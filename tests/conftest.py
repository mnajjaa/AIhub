import config_loader
import artifact_manager
import mlflow_utils
import pytest


@pytest.fixture(autouse=True)
def disable_mlflow_and_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(mlflow_utils, "MLFLOW_AVAILABLE", False, raising=False)
    monkeypatch.setattr(mlflow_utils, "mlflow", None, raising=False)

    def _fail_mlflow(self, uri, force_refresh=False):
        raise FileNotFoundError("MLflow disabled in tests")

    monkeypatch.setattr(
        artifact_manager.ArtifactManager,
        "_load_from_mlflow",
        _fail_mlflow,
        raising=False,
    )
    config_loader.set_config_loader(config_loader.ConfigLoader())
    yield
