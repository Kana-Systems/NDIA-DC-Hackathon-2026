# Local setup and verification

Run commands from the repository root in Bash on macOS/Linux, or in WSL on
Windows. Use Git LFS when cloning; a source ZIP may contain model pointers
instead of model files. Local setup installs dependencies and builds the UI;
it does not train a model or deploy AWS resources.

## 1. Check prerequisites

- Git and [Git LFS](https://git-lfs.com/).
- Python 3.12, the version used in CI. A newer Python must have compatible
  wheels for the model dependencies; use 3.12 if installation fails.
- Node.js 22.12 or newer and npm. CI uses Node 22.12.
- Internet access for dependencies and about 458 MiB of shared LFS artifacts,
  plus space for Python dependencies and the frontend build.

```bash
git --version
git lfs version
python3 --version
node --version
npm --version
```

## 2. Clone and install

```bash
git lfs install
git clone https://github.com/Kana-Systems/NDIA-DC-Hackathon-2026.git
cd NDIA-DC-Hackathon-2026
git lfs pull
bash scripts/setup-local.sh
```

If `python3` is not the desired interpreter, use
`PYTHON_BIN=python3.12 bash scripts/setup-local.sh` when creating the environment.
An existing `.venv` is reused. Setup verifies the shared artifacts, installs the
application/dev and model dependencies, runs `npm ci`, and builds `frontend/dist`.
It exits on failure; resolve the first error before launching.

## 3. Start the offline workflow

```bash
MODEL_REVIEW_ENABLED=false BEDROCK_ENABLED=false bash scripts/run-local.sh
```

Open **http://127.0.0.1:8080/lens/** and sign in with password `contract-demo`.
The script supplies local credentials and a temporary signing secret, binds only
to loopback, and uses SQLite at `artifacts/workspace/lens.sqlite`. Restarting
preserves records; a newly generated signing secret requires signing in again.
Stop with Ctrl+C. Do not delete the workspace database to update the app.

Offline mode exercises the interface, deterministic contract review, lexical
retrieval, review decisions, and exports. It does not demonstrate live Terra
generation or the hosted SageMaker ensemble. Output identifies its actual mode.

If port 8080 is occupied, add `PORT=8082` before the command and open port 8082.
Use a separate `WORKSPACE_DB_PATH` if you want an isolated demo workspace.

## 4. Verify it works

In another terminal, from the same repository:

```bash
curl --fail --silent --show-error http://127.0.0.1:8080/health
curl --fail --silent --show-error \
  --user judge:contract-demo \
  --request POST http://127.0.0.1:8080/api/v1/reviews/sample \
  --output /tmp/kana-sample-review.json
.venv/bin/python -m json.tool /tmp/kana-sample-review.json
```

Health returns `{"status":"ok"}`. The sample route returns a JSON review of a
synthetic contract; it does not insert that sample into your Lens workspace.
Use your configured password and port if you changed them. In Lens, paste a
synthetic contract, run a review, inspect a finding and its evidence, and follow
the [judging walkthrough](JUDGING_REVIEW.md) for the connected workflow.
Collections show 10 entries initially. **Show more** opens the remainder in a
bounded scroll area; **Show fewer** collapses it. Contract search covers the
whole collection, including collapsed entries.

## 5. Enable live review when ready

Configure the normal AWS credential chain with authorized GovCloud Bedrock
access. Verify it with `aws sts get-caller-identity --region us-gov-west-1`, then:

```bash
MODEL_REVIEW_ENABLED=true BEDROCK_ENABLED=true bash scripts/run-local.sh
```

This loads the selected packaged classifier and FAR/DFARS corpus and enables
GPT-5.6 Terra through Bedrock. Inference is billable and transmits submitted text
and retrieved evidence. Use public/synthetic input. An STS success confirms
identity, not Bedrock model access. Model mode reports unavailable dependencies
or services instead of silently changing to offline mode.

Advanced configuration is documented in [.env.example](../.env.example).
Pydantic reads `.env`, but exported variables take precedence. In particular,
`run-local.sh` exports its local defaults; pass mode/password overrides in the
shell when using that script. Never commit a configured `.env` or credentials.

## Update and check

After committing or stashing your own work, update your intended branch, fetch
LFS artifacts, and rerun setup:

```bash
git pull --ff-only
git lfs pull
bash scripts/setup-local.sh
bash scripts/check-local.sh
```

The check script runs Python lint/tests, deterministic benchmarks, frontend
lint/tests, and a production build. These checks use synthetic data and do not
prove live model quality. Run the separately documented live model smoke only
when its cloud access and inference cost are intended.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Artifact verification reports an LFS pointer or missing weights | Run `git lfs pull`, then `.venv/bin/python scripts/verify-shared-artifacts.py` (or `python3` before environment creation). |
| Python dependency has no compatible distribution | Create a new Python 3.12 virtual environment; select it with `PYTHON_BIN` during setup. |
| Vite reports an unsupported Node version | Use Node 22.12+ and rerun setup so `npm ci` uses that runtime. |
| `/lens/` is missing or still shows old UI | Run `npm --prefix frontend ci` and `npm --prefix frontend run build`, then restart the server. |
| Address already in use | Launch with `PORT=8082`; keep the existing server running if another task owns it. |
| Login fails | Use the script's default password or your shell override; passwords must have at least 12 characters. |
| Session expires or becomes invalid after restart | Sign in again. The UI retains open forms during session renewal. |
| AWS credentials/model access fails | Fix the configured AWS profile and Bedrock permissions, or explicitly select both offline flags. |
| Approval/export is blocked | Inspect source readiness; changed, unavailable, or unsupported evidence requires a fresh review. |
| A SharePoint file reports a sync error | Check the filename and reason shown under the connection. Temporary download failures retry up to three total attempts; persistent failures remain visible. Use **Sync now** after resolving access, size, encoding, or parsing issues. Existing records are retained, but unavailable sources cannot support decisions until a successful sync. |

SharePoint reads retry throttling, temporary server errors, timeouts, and interrupted
downloads. Retries follow Microsoft's `Retry-After` header, or use exponential
backoff when it is absent, following the
[Microsoft Graph throttling guidance](https://learn.microsoft.com/en-us/graph/throttling).
An individual retry wait is limited to 60 seconds; longer requested waits are
reported as a temporary failure rather than retried early. Interrupted downloads
restart with a fresh URL and discard partial bytes. Sync logs include the item ID,
operation stage, exception type, and HTTP status without document text or signed URLs.

For AWS hosting, use the [repository deployment instructions](../README.md#govcloud-deployment).
For current connector and production boundaries, read
[LENS_WORKSPACE.md](../LENS_WORKSPACE.md).
