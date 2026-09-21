# Backend

## Package map

The current backend lives under mirror7_backend/.

| File | Role |
|---|---|
| service.py | Orchestrates sessions, response generation, persistence, actions, and metrics. |
| runtime.py | Session runtime and realization-contract construction. |
| persistence.py | Checkpoint storage and recovery. |
| actions.py | Action gateway and policy. |
| observability.py | Metrics and status snapshots. |
| http_server.py | JSON HTTP service. |

## Session lifecycle

A client session can:

1. create a session;
2. restore a checkpoint;
3. submit observations and optional goals;
4. execute a Mirror step;
5. construct a realization contract;
6. optionally invoke a response model;
7. validate the generated response;
8. return structured state/result data;
9. save or close the session.

BackendService tracks active steps so closing a session does not race an in-flight request.

## Runtime controls

The service enforces unique session identifiers, maximum session capacity, rejection of work on closing sessions, active-step accounting, explicit errors for unknown sessions, and explicit checkpoint-configuration requirements.

## Response model lifecycle

set_response_model() installs or removes a model adapter at runtime.

The operation changes the language realization component only. It does not replace the Mirror state engine.

## HTTP API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/health | Liveness and status. |
| GET | /api/status | Detailed service status. |
| GET | /api/sessions | Active session ids. |
| GET | /api/sessions/{id} | Session snapshot. |
| POST | /api/sessions | Create session. |
| POST | /api/sessions/{id}/step | Submit observation/goal and run Mirror. |
| POST | /api/sessions/{id}/save | Save checkpoint. |
| POST | /api/sessions/{id}/restore | Restore checkpoint. |
| POST | /api/sessions/{id}/action | Invoke configured action gateway. |
| DELETE | /api/sessions/{id} | Close session. |
| DELETE | /api/sessions/{id}/checkpoint | Delete checkpoint. |

## HTTP safety boundary

The HTTP layer rejects malformed JSON, non-JSON bodies, oversized bodies, invalid session ids, invalid action arguments, unknown sessions, denied actions, and unavailable checkpoint operations.

It returns no-store cache headers and common response-hardening headers.

## Deployment

Local:

~~~bash
python -m mirror7_backend.http_server
~~~

Docker:

~~~bash
docker build -t mirror7-backend .
docker run --rm -p 8787:8787 mirror7-backend
~~~

Compose:

~~~bash
docker compose up --build
~~~

The external UI is intentionally decoupled from the backend package.

## Boundary

The backend is a software integration surface around Mirror mechanisms. It does not by itself establish model intelligence, AGI, or strong native language generation.
