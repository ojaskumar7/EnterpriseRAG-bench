# Frontend query workspace

The local frontend is a static browser client in `frontend/`. It intentionally
has no package manager or build step yet: the product surface can be exercised
while authentication, source registration, and deployment topology are still
being finalized.

## Contract

- `GET /health` drives the runtime indicator.
- `POST /query` renders the answer, grounding decision, reason, latency, and
  citation metadata.
- `GET /ingestion/jobs/{job_id}` renders the asynchronous job state.
- A scoped JWT is entered by the operator and held in browser local storage.

The API accepts browser calls from `CORS_ORIGINS`, which defaults to the local
static server at `http://localhost:4173`. Production deployments should set an
explicit allow-list and replace the manual token field with the chosen identity
provider flow.

## Run locally

```bash
python -m http.server 4173 --directory frontend
```

The frontend deliberately does not expose ingestion submission because the API
only enables registered ingestion operations. Job tracking is available now;
source upload and provider-backed authentication remain later production work.
