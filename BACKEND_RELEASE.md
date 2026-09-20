# Mirror 7 Backend Release

## Run locally

```bash
python -m mirror7_backend.http_server
```

## Configure a website origin

For local development:

```text
MIRROR7_HOST=127.0.0.1
MIRROR7_PORT=8787
MIRROR7_CORS_ORIGIN=*
```

For a deployed website, set `MIRROR7_CORS_ORIGIN` to the exact browser origin, for example `https://example.com`.

## Run tests

```bash
python -m pytest
```

## Container

```bash
docker build -t mirror7-backend .
docker run --rm -p 8787:8787 -e MIRROR7_CORS_ORIGIN=https://example.com mirror7-backend
```

## API

- GET /api/health
- GET /api/status
- GET /api/sessions
- GET /api/sessions/{id}
- POST /api/sessions
- POST /api/sessions/{id}/step
- POST /api/sessions/{id}/save
- POST /api/sessions/{id}/restore
- POST /api/sessions/{id}/action
- DELETE /api/sessions/{id}/checkpoint
- DELETE /api/sessions/{id}

The HTTP boundary includes JSON body limits, content-type enforcement, browser CORS configuration, security response headers, session lifecycle controls, checkpoint lifecycle controls, and allow-listed action execution through the backend service.

The cognition engine remains injected; this release packages the backend boundary and does not claim AGI.
