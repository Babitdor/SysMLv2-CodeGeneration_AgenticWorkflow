// Local fallback when /api/validate cannot be reached: structural checks only.
// Deliberately narrow - it reports unbalanced delimiters and unterminated
// strings, never a grammar verdict it cannot actually make.
import type { Diagnostic } from './types';

export interface StructuralReport {
  ok: boolean;
  diagnostics: Diagnostic[];
  summary: string;
}

const DECLARATION = /^\s*(part|item|attribute|port|action|state|constraint|interface|connection|requirement)\s+def\b[^;{}]*$/;
const EMPTY_PACKAGE = /package\s+[\w'" ]*\{\s*\}/;

export function structuralCheck(code: string): StructuralReport {
  const diagnostics: Diagnostic[] = [];
  const lines = code.split('\n');
  if (!code.trim()) {
    return {
      ok: false,
      diagnostics: [{ severity: 'error', line: 1, column: 0, message: 'the model is empty' }],
      summary: 'No model text to check.',
    };
  }

  let braces = 0;
  let parens = 0;
  let inBlockComment = false;

  lines.forEach((raw, index) => {
    const line = index + 1;
    let quote: string | null = null;
    for (let i = 0; i < raw.length; i += 1) {
      const ch = raw[i];
      if (inBlockComment) {
        if (ch === '*' && raw[i + 1] === '/') {
          inBlockComment = false;
          i += 1;
        }
        continue;
      }
      if (quote) {
        if (ch === '\\') i += 1;
        else if (ch === quote) quote = null;
        continue;
      }
      if (ch === '/' && raw[i + 1] === '/') break;
      if (ch === '/' && raw[i + 1] === '*') {
        inBlockComment = true;
        i += 1;
        continue;
      }
      if (ch === String.fromCharCode(34) || ch === String.fromCharCode(39)) {
        quote = ch;
        continue;
      }
      if (ch === '{') braces += 1;
      else if (ch === '}') braces -= 1;
      else if (ch === '(') parens += 1;
      else if (ch === ')') parens -= 1;
    }

    if (quote) {
      diagnostics.push({ severity: 'error', line, column: raw.length, message: 'unterminated string literal' });
    }
    if (braces < 0) {
      diagnostics.push({ severity: 'error', line, column: 0, message: 'a closing brace has no matching open' });
      braces = 0;
    }
    if (parens < 0) {
      diagnostics.push({ severity: 'error', line, column: 0, message: 'a closing parenthesis has no matching open' });
      parens = 0;
    }
    if (DECLARATION.test(raw)) {
      diagnostics.push({ severity: 'hint', line, column: raw.trimEnd().length, message: 'declaration has no terminator or body' });
    }
  });

  if (inBlockComment) {
    diagnostics.push({ severity: 'error', line: lines.length, column: 0, message: 'block comment is never closed' });
  }
  if (braces > 0) {
    diagnostics.push({ severity: 'error', line: lines.length, column: 0, message: braces + ' unclosed opening brace at end of file' });
  }
  if (parens > 0) {
    diagnostics.push({ severity: 'error', line: lines.length, column: 0, message: parens + ' unclosed opening parenthesis at end of file' });
  }
  if (EMPTY_PACKAGE.test(code)) {
    diagnostics.push({ severity: 'hint', line: 1, column: 0, message: 'package body is empty' });
  }

  const errors = diagnostics.filter((d) => d.severity === 'error');
  return {
    ok: errors.length === 0,
    diagnostics,
    summary: errors.length === 0
      ? 'Structural checks passed (grammar not verified).'
      : 'Structural checks found ' + errors.length + ' problem(s).',
  };
}
