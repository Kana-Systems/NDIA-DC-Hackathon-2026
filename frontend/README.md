# Acquisition Lens frontend

A React/TypeScript interface for federal contract review, connected to the repository's
FastAPI backend. It keeps the bearer token in memory and displays actual model,
source, and clause-status information from the review report.

## Run with the backend

Prerequisites: Node.js 20+ and the backend listening on `http://127.0.0.1:8080`.

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/lens/`. Vite proxies `/api` to port 8080. The built
bundle is served directly at `http://127.0.0.1:8080/lens/`. Sign in using the
backend's `GRADIO_PASSWORD` (`contract-demo` with the local run script).

To point at another API, set `VITE_API_BASE_URL` in `.env`.

## Explicit offline presentation mode

Offline mock behavior is disabled by default. To intentionally run a deterministic, clearly labeled presentation without the backend, set both values in `.env`:

```dotenv
VITE_ENABLE_DEMO_MODE=true
VITE_DEMO_PASSWORD=choose-a-local-presentation-password
```

Use that configured password at the login screen. The amber banner remains visible throughout the offline presentation. Never ship a production build with offline presentation mode enabled.

## Checks

```bash
npm run lint
npm test
npm run build
```

The production output is written to `dist/`. The UI is responsive and keyboard accessible, respects reduced-motion preferences, exposes errors as live alerts, labels confidence and citation verification status, and keeps the human-review/not-legal-advice warning visible in the application footer.

## API integration

- `POST /api/auth/login`: stores `access_token` only in module memory.
- `POST /api/analyze`: sends `Authorization: Bearer <access_token>` and explicitly maps backend snake_case fields into view models.
- `POST /api/review-upload`: submits PDF/DOCX and acquisition metadata as multipart data.
- `GET /api/demo/sample`: loads synthetic contract text and matching acquisition metadata.
- `GET /api/sources`: sends the same bearer token and maps the backend source registry into catalog cards.

No contract text, token, or password is persisted in browser storage.
