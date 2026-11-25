// Dependency-free markdown: the renderer ported from the old chat page, but
// returning data instead of HTML. React escapes text nodes, so no string of
// markup ever reaches innerHTML and there is no XSS surface to audit.

export type MdInline =
  | { kind: 'text'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'strong'; text: string }
  | { kind: 'em'; text: string }
  | { kind: 'link'; text: string; href: string };

export type MdBlock =
  | { kind: 'code'; lang: string; code: string }
  | { kind: 'heading'; level: 1 | 2 | 3 | 4; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] }
  | { kind: 'quote'; text: string }
  | { kind: 'hr' };

export type TokKind = 'text' | 'com' | 'str' | 'num' | 'key' | 'type';
export interface Tok {
  kind: TokKind;
  text: string;
}

const FENCE = /^```([\w+#.-]*)\s*$/;
const FENCE_END = /^```\s*$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const QUOTE = /^>\s?/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const ORDERED = /^\s*\d+[.)]\s+(.*)$/;

/** Only http(s) links are rendered; everything else stays plain text. */
function safeHref(href: string): string {
  const raw = String(href || '').trim();
  return /^https?:\/\//i.test(raw) ? raw : '';
}

export function parseMarkdown(src: string): MdBlock[] {
  const text = String(src ?? '').replace(/\r\n?/g, '\n');
  const lines = text.split('\n');
  const out: MdBlock[] = [];
  let paragraph: string[] = [];
  let index = 0;

  const flush = () => {
    if (!paragraph.length) return;
    out.push({ kind: 'paragraph', text: paragraph.join(' ') });
    paragraph = [];
  };

  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(FENCE);

    if (fence) {
      flush();
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !FENCE_END.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // the closing fence
      out.push({ kind: 'code', lang: fence[1].toLowerCase(), code: body.join('\n') });
      continue;
    }

    if (!line.trim()) {
      flush();
      index += 1;
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length, 4) as 1 | 2 | 3 | 4;
      out.push({ kind: 'heading', level, text: heading[2].trim() });
      index += 1;
      continue;
    }

    if (RULE.test(line)) {
      flush();
      out.push({ kind: 'hr' });
      index += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      flush();
      const quoted: string[] = [];
      while (index < lines.length && QUOTE.test(lines[index])) {
        quoted.push(lines[index].replace(QUOTE, ''));
        index += 1;
      }
      out.push({ kind: 'quote', text: quoted.join(' ') });
      continue;
    }

    const bullet = line.match(BULLET);
    if (bullet) {
      flush();
      const items: string[] = [];
      for (;;) {
        const item = index < lines.length ? lines[index].match(BULLET) : null;
        if (!item) break;
        items.push(item[1]);
        index += 1;
      }
      out.push({ kind: 'list', ordered: false, items });
      continue;
    }

    if (ORDERED.test(line)) {
      flush();
      const items: string[] = [];
      for (;;) {
        const item = index < lines.length ? lines[index].match(ORDERED) : null;
        if (!item) break;
        items.push(item[1]);
        index += 1;
      }
      out.push({ kind: 'list', ordered: true, items });
      continue;
    }

    paragraph.push(line.trim());
    index += 1;
  }

  flush();
  return out;
}

const URL_START = /^https?:\/\/[^\s<>&]+/;

/** code, links, bold, italic — in that precedence, like the old renderer. */
export function parseInline(src: string): MdInline[] {
  const text = String(src ?? '');
  const out: MdInline[] = [];
  let buffer = '';
  let index = 0;

  const flush = () => {
    if (!buffer) return;
    out.push({ kind: 'text', text: buffer });
    buffer = '';
  };

  while (index < text.length) {
    const ch = text[index];

    if (ch === '`') {
      const end = text.indexOf('`', index + 1);
      const inner = end === -1 ? '' : text.slice(index + 1, end);
      if (inner && !inner.includes('\n')) {
        flush();
        out.push({ kind: 'code', text: inner });
        index = end + 1;
        continue;
      }
    }

    if (ch === '[') {
      const link = /^\[([^\]]*)\]\(([^)\s]+)\)/.exec(text.slice(index));
      const href = link ? safeHref(link[2]) : '';
      if (link && href) {
        flush();
        out.push({ kind: 'link', text: link[1], href });
        index += link[0].length;
        continue;
      }
    }

    if (ch === '*' && text[index + 1] === '*') {
      const end = text.indexOf('**', index + 2);
      const inner = end === -1 ? '' : text.slice(index + 2, end);
      if (inner && !inner.includes('\n')) {
        flush();
        out.push({ kind: 'strong', text: inner });
        index = end + 2;
        continue;
      }
    }

    if (ch === '*') {
      const end = text.indexOf('*', index + 1);
      const inner = end === -1 ? '' : text.slice(index + 1, end);
      if (inner && !inner.includes('\n') && !inner.startsWith(' ')) {
        flush();
        out.push({ kind: 'em', text: inner });
        index = end + 1;
        continue;
      }
    }

    if (ch === '_') {
      const before = index === 0 ? ' ' : text[index - 1];
      const end = text.indexOf('_', index + 1);
      const inner = end === -1 ? '' : text.slice(index + 1, end);
      const after = end === -1 ? '' : (text[end + 1] ?? '');
      const closed = after === '' || /[\s.,;:!?)]/.test(after);
      if (/[\s(]/.test(before) && inner && !inner.includes('\n') && closed) {
        flush();
        out.push({ kind: 'em', text: inner });
        index = end + 1;
        continue;
      }
    }

    if (ch === 'h') {
      const url = URL_START.exec(text.slice(index));
      if (url) {
        flush();
        out.push({ kind: 'link', text: url[0], href: url[0] });
        index += url[0].length;
        continue;
      }
    }

    buffer += ch;
    index += 1;
  }

  flush();
  return out;
}

const WORDS: Record<string, string> = {
  sysml:
    // Every keyword literal in tools/sysml_parser/LoggingParsers/SysMLv2/SysMLv2.g4.
    'about abstract accept action actor after alias all allocate allocation analysis and as assert assign assume at attribute bind binding by calc case comment concern connect connection constraint decide def default defined dependency derived do doc else end entry enum event exhibit exit expose false filter first flow for fork frame from hastype if implies import in include individual inout interface istype item join language library locale loop merge message meta metadata nonunique not null objective occurrence of or ordered out package parallel part perform port private protected public readonly redefines ref references render rendering rep require requirement return satisfy send snapshot specializes stakeholder standard state subject subsets succession then timeslice to transition true until use variant variation verification verify via view viewpoint when while xor',
  python:
    'and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield True False None self match case',
  javascript:
    'as async await break case catch class const continue debugger default delete do else export extends false finally for from function if import in instanceof let new null of return static super switch this throw true try typeof undefined var void while with yield',
  json: 'true false null',
  bash: 'if then else elif fi for while do done case esac function return in export local echo cd set unset',
  yaml: 'true false null yes no on off',
};

interface Spec {
  line: string;
  hash: boolean;
  quotes: string;
  words: string;
  /** SysML: names after a declaration keyword, and after ":" or "@". */
  names?: boolean;
  /** SysML: C-style block comments as well as line comments. */
  block?: boolean;
}

const QUOTES = '"\'`';
const PLAIN: Spec = { line: '', hash: false, quotes: '', words: '' };

const SPECS: Record<string, Spec> = {
  sysml: { line: '//', hash: true, quotes: '"\'', words: 'sysml', names: true, block: true },
  sysmlv2: { line: '//', hash: true, quotes: '"\'', words: 'sysml', names: true, block: true },
  python: { line: '', hash: true, quotes: '"\'', words: 'python' },
  py: { line: '', hash: true, quotes: '"\'', words: 'python' },
  javascript: { line: '//', hash: false, quotes: QUOTES, words: 'javascript' },
  js: { line: '//', hash: false, quotes: QUOTES, words: 'javascript' },
  typescript: { line: '//', hash: false, quotes: QUOTES, words: 'javascript' },
  ts: { line: '//', hash: false, quotes: QUOTES, words: 'javascript' },
  json: { line: '', hash: false, quotes: '"', words: 'json' },
  bash: { line: '', hash: true, quotes: '"\'', words: 'bash' },
  sh: { line: '', hash: true, quotes: '"\'', words: 'bash' },
  shell: { line: '', hash: true, quotes: '"\'', words: 'bash' },
  yaml: { line: '', hash: true, quotes: '"\'', words: 'yaml' },
  yml: { line: '', hash: true, quotes: '"\'', words: 'yaml' },
  text: PLAIN,
};

/** Keywords whose next word is the declaration's own name. */
const DECLARES = new Set(['def', 'package', 'alias', 'import', 'metadatadef']);

const wordCache = new Map<string, Set<string>>();

function wordSet(name: string): Set<string> {
  if (!name) return new Set();
  let set = wordCache.get(name);
  if (!set) {
    set = new Set((WORDS[name] ?? '').split(' ').filter(Boolean));
    wordCache.set(name, set);
  }
  return set;
}

const NUMBERISH = /[0-9a-fA-FxX._]/;
const WORD_CHAR = /[A-Za-z_]/;
const WORD_REST = /[A-Za-z0-9_]/;

/** Tokenizer for the code blocks; unknown languages fall through as text. */
export function highlight(code: string, lang: string): Tok[] {
  const spec = SPECS[String(lang || '').toLowerCase()] ?? PLAIN;
  const words = wordSet(spec.words);
  const src = String(code ?? '');
  const out: Tok[] = [];
  let index = 0;

  const push = (kind: TokKind, text: string) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text;
    else out.push({ kind, text });
  };

  const lineEnd = (from: number) => {
    const stop = src.indexOf('\n', from);
    return stop === -1 ? src.length : stop;
  };

  /** The identifier after `from`, skipping whitespace; null when there is none. */
  const readWord = (from: number) => {
    let at = from;
    while (at < src.length && /\s/.test(src[at])) at += 1;
    if (at >= src.length || !WORD_CHAR.test(src[at])) return null;
    let end = at + 1;
    while (end < src.length && WORD_REST.test(src[end])) end += 1;
    return { word: src.slice(at, end), start: at, end };
  };

  while (index < src.length) {
    const ch = src[index];

    if (spec.block && src.startsWith('/*', index)) {
      const stop = src.indexOf('*/', index + 2);
      const end = stop === -1 ? src.length : stop + 2;
      push('com', src.slice(index, end));
      index = end;
      continue;
    }

    if (spec.line && src.startsWith(spec.line, index)) {
      const stop = lineEnd(index);
      push('com', src.slice(index, stop));
      index = stop;
      continue;
    }

    if (spec.hash && ch === '#') {
      const stop = lineEnd(index);
      push('com', src.slice(index, stop));
      index = stop;
      continue;
    }

    if (spec.quotes.includes(ch)) {
      let end = index + 1;
      while (end < src.length) {
        const next = src[end];
        if (next === '\\') {
          end += 2;
          continue;
        }
        if (next === ch) {
          end += 1;
          break;
        }
        if (next === '\n' && ch !== '`') break;
        end += 1;
      }
      push('str', src.slice(index, end));
      index = end;
      continue;
    }

    if (ch >= '0' && ch <= '9') {
      let end = index + 1;
      while (end < src.length && NUMBERISH.test(src[end])) end += 1;
      push('num', src.slice(index, end));
      index = end;
      continue;
    }

    if (WORD_CHAR.test(ch)) {
      let end = index + 1;
      while (end < src.length && WORD_REST.test(src[end])) end += 1;
      const word = src.slice(index, end);
      push(words.has(word) ? 'key' : 'text', word);
      index = end;
      // `part def Engine` / `package Pkg`: the next word names the declaration.
      if (spec.names && DECLARES.has(word)) {
        const name = readWord(index);
        if (name) {
          push('text', src.slice(index, name.start));
          push('type', name.word);
          index = name.end;
        }
      }
      continue;
    }

    // `:>, :>>, ::` and `@` are followed by a type or metadata reference.
    if (spec.names && (ch === ':' || ch === '@')) {
      let end = index + 1;
      while (end < src.length && (src[end] === '>' || src[end] === ':')) end += 1;
      push('text', src.slice(index, end));
      const name = readWord(end);
      if (name) {
        push('text', src.slice(end, name.start));
        push('type', name.word);
        index = name.end;
      } else {
        index = end;
      }
      continue;
    }

    push('text', ch);
    index += 1;
  }

  return out;
}

/** Token kind → CSS class; shared by transcript code blocks and the output pane. */
export const TOKEN_CLASS: Record<string, string> = {
  com: 'tok-com',
  str: 'tok-str',
  num: 'tok-num',
  key: 'tok-key',
  type: 'tok-type',
};
