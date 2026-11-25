# Frontend

React 19 + Vite + Tailwind chat UI for the SysML v2 workflow. It is one of the
two components in this repository; the other is the FastAPI backend in
`../webapp`.

## Where it is served

The build writes to `../webapp/static/ui`, which `webapp/server.py` mounts at
`/static` and reads `index.html` from. Nothing else serves this app: in
production the bundle is baked into the Python image by the `ui` stage of
`../Dockerfile`, so one container answers both `/` and `/api`.

## Running it

```bash
docker compose up -d frontend   # from the repository root: http://localhost:5173
```

The `frontend` service in `docker-compose.yml` runs `npm run dev` with the
source bind-mounted, so edits hot-reload. It proxies `/api` to the app container
(`VITE_PROXY_TARGET`), which is what makes sign-in and streaming work on 5173.

Without Docker, against a backend on 127.0.0.1:8000:

```bash
npm ci
npm run dev        # http://localhost:5173
npm run typecheck  # tsc -b
npm run lint       # oxlint
npm run build      # writes ../webapp/static/ui
```

## Layout

- `src/components/` — one file per surface (Sidebar, Composer, ThreadView, ...)
- `src/state/WorkspaceContext.tsx` — the single store the surfaces read
- `src/lib/` — transport and pure helpers (`api.ts`, `sse.ts`, `markdown.ts`)
- `src/index.css` — Tailwind entry and the design tokens

## node_modules

The `frontend` service keeps `node_modules` in a named volume. A bind mount of
the source would otherwise expose the host install, whose esbuild/rollup binaries
are built for the host platform and cannot load in the Linux image.
