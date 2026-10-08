from pathlib import Path


def test_dockerfile_uses_non_root_runtime_and_healthcheck() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "USER mlforge" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    assert "mlforge.serving.app:app" in dockerfile
    assert '--host", "0.0.0.0"' in dockerfile
    assert '--port", "8000"' in dockerfile


def test_dockerignore_excludes_local_runtime_state() -> None:
    ignored = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())

    required = {
        ".git",
        ".venv",
        ".env",
        "data",
        "notebooks",
        "tests",
        "mlruns",
        "mlartifacts",
        "*.db",
    }
    assert required <= ignored
