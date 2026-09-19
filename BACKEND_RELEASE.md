# Mirror 7 Backend Release

## Run locally
python -m mirror7_backend.http_server

## Run tests
python -m pytest

## Container
docker build -t mirror7-backend .
docker run --rm -p 8787:8787 mirror7-backend

## API
- GET /api/health
- GET /api/status
- GET /api/sessions
- GET /api/sessions/{id}
- POST /api/sessions
- POST /api/sessions/{id}/step
- POST /api/sessions/{id}/save
- POST /api/sessions/{id}/restore
- DELETE /api/sessions/{id}/checkpoint
- DELETE /api/sessions/{id}

The cognition engine remains injected; this release packages the backend boundary and does not claim AGI.
