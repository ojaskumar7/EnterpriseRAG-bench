# Redwood Knowledge Intelligence UI

Dependency-free local frontend for the authenticated FastAPI service.

```bash
python -m http.server 4173 --directory frontend
```

Open <http://localhost:4173>. In **Connection → edit**, set the API URL and paste
a JWT containing `sub`, `tenant_id`, `knowledge_base_ids`, and `exp`. The token is
stored only in this browser's local storage; the frontend does not issue tokens.

The UI uses `POST /query`, `GET /health`, and `GET /ingestion/jobs/{job_id}`.
