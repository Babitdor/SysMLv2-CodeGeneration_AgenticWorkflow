// One validation entry point for the UI. Server grammar first, local
// structural checks when it cannot be reached.
import { validateCode } from './api';
import { structuralCheck } from './structural';
import type { ValidationResult } from './types';

export async function validateSysML(code: string, attempt = 0): Promise<ValidationResult> {
  try {
    const report = await validateCode(code);
    const status = report.status === 'invalid' ? 'invalid' : report.status === 'valid' ? 'valid' : 'error';
    return {
      status,
      ok: report.ok,
      attempt,
      final: report.ok,
      diagnostics: report.diagnostics ?? [],
      summary: report.summary ?? '',
      detail: report.detail,
      source: 'antlr',
    };
  } catch (err) {
    const local = structuralCheck(code);
    const reason = err instanceof Error ? err.message : String(err);
    return {
      status: local.ok ? 'valid' : 'invalid',
      ok: local.ok,
      attempt,
      final: local.ok,
      diagnostics: local.diagnostics,
      summary: local.summary + ' Server unavailable: ' + reason,
      source: 'structural',
    };
  }
}
