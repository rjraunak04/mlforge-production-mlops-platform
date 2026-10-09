# MLForge Release Checklist

A release candidate is ready for `main` only when all applicable items are satisfied.

- [ ] Branch is based on the latest `develop`.
- [ ] Ruff lint passes.
- [ ] Ruff format check passes.
- [ ] Full pytest suite passes.
- [ ] Source compilation passes.
- [ ] Linux Docker build passes in GitHub Actions.
- [ ] No raw dataset, local MLflow database, model artifact, secret, or production log is committed.
- [ ] README describes the actual implemented architecture.
- [ ] Known production boundaries are documented rather than hidden.
- [ ] Model promotion still requires quality gates and strict champion improvement.
- [ ] Release PR targets `main` from the validated release state.
- [ ] Release tag is created only after the release PR is merged.
