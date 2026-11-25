// Client for POST /api/model/index and /api/model/impact: the parse-tree index
// the Model panel reads. Both take the code, so a panel opened on an old turn
// indexes that turn's model and nothing else.

export interface ModelElement {
  id: string;
  name: string;
  kind: string;
  qualified_name: string;
  parent: string | null;
  line: number;
  column: number;
  end_line: number;
}

export interface ModelEdge {
  source: string;
  target: string;
  kind: string;
  label: string;
  line: number;
  resolved: boolean;
}

export interface ModelIssue {
  severity: 'error' | 'warning' | 'info';
  message: string;
  line: number;
  column: number;
  rule: string;
  element: string | null;
}

export interface ModelStats {
  elements: number;
  edges: number;
  issues: number;
  unresolved: number;
  requirements: number;
}

/** One traceability or impact row: an element, plus the edge that reached it. */
export interface ModelRow {
  id: string;
  name: string;
  kind: string;
  line: number;
  satisfied?: boolean;
  via?: string;
}

export interface Traceability {
  requirements: ModelRow[];
  unlinked: ModelRow[];
  counts: { requirements: number; satisfied: number; unlinked: number };
}

export interface ModelReport {
  revision: string;
  /** antlr, or why there is no index: syntax-error, empty, unavailable. */
  validator: string;
  elements: ModelElement[];
  edges: ModelEdge[];
  issues: ModelIssue[];
  stats: ModelStats;
  truncated: boolean;
  review: ModelIssue[];
  traceability: Traceability;
}

export interface ImpactReport {
  direct: ModelRow[];
  transitive: ModelRow[];
  requirements: ModelRow[];
  truncated: boolean;
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    // The server's detail is the only explanation of a 400 worth showing.
    let detail = String(res.status);
    try {
      const payload = (await res.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      /* non-JSON error body: the status is all we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function loadModelIndex(
  code: string,
  signal?: AbortSignal,
): Promise<ModelReport> {
  return post<ModelReport>('/api/model/index', { code }, signal);
}

export async function analyzeImpact(
  code: string,
  elementId: string,
  signal?: AbortSignal,
): Promise<ImpactReport> {
  return post<ImpactReport>(
    '/api/model/impact',
    { code, element_id: elementId },
    signal,
  );
}
