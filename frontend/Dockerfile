# The frontend, as its own component.
#
# Only a development target exists here. Production does not run this image: the
# static bundle is built in the `ui` stage of ../Dockerfile and baked into the
# Python image, so there is one container serving both / and /api and nothing to
# keep in sync at deploy time.
#
#   docker compose up -d frontend     # Vite + HMR on http://localhost:5173
#
# The dependencies installed here are the ones that run: a bind mount of the
# source tree would otherwise expose the host's node_modules, which holds
# platform-specific esbuild/rollup binaries that cannot load in Linux. Compose
# keeps this directory in a named volume instead (see docker-compose.yml).
FROM node:22-alpine AS dev
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
EXPOSE 5173
# --host and the proxy target come from vite.config.ts.
CMD ["npm", "run", "dev"]
