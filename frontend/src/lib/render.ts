// Client for POST /api/render: the kernel's %viz drawing of a model, returned
// as a data-URL SVG. A render takes seconds to minutes and the kernel has few
// slots, so callers render on demand and pass a signal to cancel.

export type Orientation = 'TB' | 'LR';
export type Compartments = 'default' | 'compact' | 'tree';

export interface RenderOptions {
  /** Empty renders the model's root package. */
  element?: string;
  orientation?: Orientation;
  compartments?: Compartments;
}

export interface RenderReply {
  success: boolean;
  data_url?: string;
  /** The service's own explanation when `success` is false. */
  message?: string;
  cached?: boolean;
}

export async function renderModel(
  code: string,
  options: RenderOptions = {},
  signal?: AbortSignal,
): Promise<RenderReply> {
  const res = await fetch('/api/render', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      code,
      element_name: options.element ?? '',
      orientation: options.orientation ?? 'TB',
      compartments: options.compartments ?? 'default',
    }),
    signal,
  });
  if (!res.ok) {
    // 409 means "no model yet"; the server's detail says so in words.
    let detail = String(res.status);
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body: the status is all we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as RenderReply;
}
