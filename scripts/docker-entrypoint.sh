#!/bin/sh
# Start gunicorn with SYSML_WORKERS workers, or run whatever was asked for.
#
# gunicorn's --workers takes a number and CMD cannot expand a variable, which is
# the whole reason this script exists. Any arguments are executed instead, so
# "docker compose run app python -m webapp.db --check" still works.
set -eu

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec gunicorn webapp.server:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "${SYSML_WORKERS:-2}" \
    --bind "0.0.0.0:${PORT:-8000}" \
    --timeout "${SYSML_GUNICORN_TIMEOUT:-120}" \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile -
