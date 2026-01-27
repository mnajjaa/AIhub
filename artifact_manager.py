"""
MLflow Artifact Manager - Unified artifact loading from MLflow and local filesystem
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict
import hashlib

try:
    from mlflow import MlflowClient
    from mlflow.artifacts import download_artifacts
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MlflowClient = None
    download_artifacts = None

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages artifact loading from MLflow and local filesystem with caching"""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        cache_dir: str = ".artifact_cache",
        run_id_map: Optional[Dict[str, str]] = None,
        artifact_uri_map: Optional[Dict[str, str]] = None,
        default_run_id: Optional[str] = None,
        fallback_paths: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize ArtifactManager

        Args:
            tracking_uri: MLflow tracking server URI (default: sqlite:///mlflow.db)
            registry_uri: MLflow registry URI (default: MLFLOW_REGISTRY_URI or tracking_uri)
            cache_dir: Directory for caching downloaded artifacts
            run_id_map: Map of collection/item to run IDs
            artifact_uri_map: Map of collection/item/version to full MLflow artifact URIs
            default_run_id: Fallback run ID for all artifacts
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///mlflow.db"
        )
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI") or self.tracking_uri
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.run_id_map = run_id_map or {}
        self.artifact_uri_map = artifact_uri_map or {}
        self.default_run_id = default_run_id
        self.fallback_paths = fallback_paths or {}
        self.extension_map = {
            "prompts": ".txt",
            "schemas": ".json",
            "configs": ".yaml",
            "config": ".yaml",
        }

        # Initialize MLflow client if available
        self.client = None
        self._init_client()

        logger.info(f"ArtifactManager initialized with tracking_uri: {self.tracking_uri}")

    def _init_client(self):
        """Initialize MLflow client if available."""
        self.client = None
        if MLFLOW_AVAILABLE:
            try:
                self.client = MlflowClient(
                    tracking_uri=self.tracking_uri,
                    registry_uri=self.registry_uri
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize MLflow client: {e}. Fallback to local files only."
                )
        else:
            logger.warning(
                "MLflow not installed. Only local file loading supported. "
                "Install MLflow to enable artifact tracking."
            )

    def configure(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        run_id_map: Optional[Dict[str, str]] = None,
        artifact_uri_map: Optional[Dict[str, str]] = None,
        default_run_id: Optional[str] = None,
        fallback_paths: Optional[Dict[str, str]] = None,
    ):
        """
        Update MLflow configuration and mappings.

        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow registry URI
            run_id_map: Map of collection/item to run IDs
            artifact_uri_map: Map of collection/item/version to full MLflow artifact URIs
            default_run_id: Fallback run ID for all artifacts
        """
        if tracking_uri:
            self.tracking_uri = tracking_uri
        if registry_uri:
            self.registry_uri = registry_uri
        if run_id_map:
            self.run_id_map.update(run_id_map)
        if artifact_uri_map:
            self.artifact_uri_map.update(artifact_uri_map)
        if default_run_id:
            self.default_run_id = default_run_id
        if fallback_paths:
            self.fallback_paths.update(fallback_paths)

        if tracking_uri or registry_uri:
            self._init_client()
    
    def _is_mlflow_uri(self, path: str) -> bool:
        """Check if path is an MLflow URI"""
        return path.startswith("mlflow://")
    
    def _parse_mlflow_uri(self, uri: str) -> tuple[str, str, str]:
        """
        Parse MLflow URI format: mlflow://collection/item/version
        
        Args:
            uri: MLflow URI like "mlflow://prompts/summarizer/system_v1"
            
        Returns:
            Tuple of (collection, item, version)
        """
        # Remove mlflow:// prefix
        prefix = "mlflow://"
        if not uri.startswith(prefix):
            raise ValueError(f"Invalid MLflow URI format: {uri}. Expected {prefix}collection/item/version")
        path = uri[len(prefix):]
        parts = path.split("/")
        
        if len(parts) < 3:
            raise ValueError(f"Invalid MLflow URI format: {uri}. Expected mlflow://collection/item/version")
        
        collection = parts[0]
        item = parts[1]
        version = parts[2]
        
        return collection, item, version
    
    def _get_cache_path(self, uri: str) -> Path:
        """Generate cache file path from URI"""
        uri_hash = hashlib.md5(uri.encode()).hexdigest()
        return self.cache_dir / uri_hash
    
    def load_artifact(self, path: str, force_refresh: bool = False) -> str:
        """
        Load artifact content from MLflow or local filesystem
        
        Args:
            path: Either local file path or mlflow://collection/item/version URI
            force_refresh: Force re-download from MLflow (bypass cache)
            
        Returns:
            Artifact content as string
            
        Raises:
            FileNotFoundError: If artifact not found
            ValueError: If MLflow URI is invalid
        """
        if self._is_mlflow_uri(path):
            try:
                return self._load_from_mlflow(path, force_refresh)
            except Exception as e:
                fallback = self.fallback_paths.get(path)
                if fallback:
                    logger.warning(
                        f"Falling back to local path for {path} due to MLflow error: {e}"
                    )
                    return self._load_from_local(fallback)
                raise
        return self._load_from_local(path)

    def _load_from_mlflow(self, uri: str, force_refresh: bool = False) -> str:
        """Load artifact from MLflow"""
        if self.client is None:
            raise RuntimeError("MLflow client not initialized. Cannot load from MLflow URI.")
        
        # Check cache first
        cache_path = self._get_cache_path(uri)
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading artifact from cache: {uri}")
            with open(cache_path, "r") as f:
                return f.read()
        
        try:
            collection, item, version = self._parse_mlflow_uri(uri)

            artifact_uri = self._resolve_artifact_uri(collection, item, version)

            logger.info(f"Downloading artifact from MLflow: {uri} -> {artifact_uri}")

            # Download artifact
            local_path = download_artifacts(
                artifact_uri=artifact_uri,
                tracking_uri=self.tracking_uri,
                dst_path=str(self.cache_dir)
            )
            
            # Read content
            artifact_file_path = Path(local_path)
            if artifact_file_path.is_dir():
                # If directory, look for a file matching item name
                for file in artifact_file_path.glob(f"{item}*"):
                    if file.is_file():
                        with open(file, "r") as f:
                            content = f.read()
                        # Cache the content
                        with open(cache_path, "w") as f:
                            f.write(content)
                        logger.info(f"Successfully loaded artifact from MLflow: {uri}")
                        return content
                raise FileNotFoundError(f"No matching artifact file found for {item}")
            else:
                # Single file
                with open(artifact_file_path, "r") as f:
                    content = f.read()
                # Cache the content
                with open(cache_path, "w") as f:
                    f.write(content)
                logger.info(f"Successfully loaded artifact from MLflow: {uri}")
                return content
                
        except Exception as e:
            logger.error(f"Failed to load artifact from MLflow: {uri} - {e}")
            raise FileNotFoundError(f"Failed to load MLflow artifact {uri}: {e}")

    def _resolve_artifact_uri(self, collection: str, item: str, version: str) -> str:
        """Resolve mlflow://collection/item/version to a concrete artifact URI."""
        logical_key = f"{collection}/{item}/{version}"
        if logical_key in self.artifact_uri_map:
            return self.artifact_uri_map[logical_key]
        prefixed_key = f"mlflow://{logical_key}"
        if prefixed_key in self.artifact_uri_map:
            return self.artifact_uri_map[prefixed_key]

        run_id = self._lookup_run_id(collection, item)
        if run_id:
            artifact_path = self._build_artifact_path(collection, item, version)
            return f"runs:/{run_id}/{artifact_path}"

        # Backward-compatible fallback: treat version as run_id if it looks like one
        if self._looks_like_run_id(version):
            artifact_path = f"{collection}/{item}"
            return f"runs:/{version}/{artifact_path}"

        # Registry fallback: models:/<collection>_<item>/<version|alias|stage>
        return self._resolve_registry_uri(collection, item, version)

    def _build_artifact_path(self, collection: str, item: str, version: str) -> str:
        """Construct artifact path within a run from logical MLflow URI pieces."""
        filename = version
        if "." not in filename:
            extension = self.extension_map.get(collection, "")
            if extension:
                filename = f"{filename}{extension}"
        return f"{collection}/{item}/{filename}"

    def _lookup_run_id(self, collection: str, item: str) -> Optional[str]:
        """Find run ID for collection/item based on configured mappings."""
        direct_key = f"{collection}/{item}"
        if direct_key in self.run_id_map:
            return self.run_id_map[direct_key]
        if item in self.run_id_map:
            return self.run_id_map[item]
        if collection in self.run_id_map:
            return self.run_id_map[collection]
        if "default" in self.run_id_map:
            return self.run_id_map["default"]
        return self.default_run_id

    def _looks_like_run_id(self, value: str) -> bool:
        """Check if a string resembles an MLflow run ID (UUID-like)."""
        if not isinstance(value, str):
            return False
        trimmed = value.replace("-", "")
        if len(trimmed) != 32:
            return False
        return all(c in "0123456789abcdef" for c in trimmed.lower())

    def _resolve_registry_uri(self, collection: str, item: str, version: str) -> str:
        """Resolve artifact via MLflow Model Registry."""
        if self.client is None:
            raise RuntimeError("MLflow client not initialized. Cannot resolve registry URIs.")

        model_name = f"{collection}_{item}"
        try:
            if version.isdigit():
                mv = self.client.get_model_version(model_name, version=version)
                return mv.source
        except Exception:
            pass

        # Try alias if supported
        try:
            mv = self.client.get_model_version_by_alias(model_name, version)
            return mv.source
        except Exception:
            pass

        # Try stage name (Production/Staging/etc.)
        try:
            latest = self.client.get_latest_versions(model_name, stages=[version])
            if latest:
                return latest[0].source
        except Exception:
            pass

        # Final fallback to models:/ URI
        return f"models:/{model_name}/{version}"
    
    def _load_from_local(self, path: str) -> str:
        """Load artifact from local filesystem"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
        
        try:
            with open(path, "r") as f:
                content = f.read()
            logger.debug(f"Loaded artifact from local filesystem: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to load local artifact {path}: {e}")
            raise
    
    def load_json_artifact(self, path: str, force_refresh: bool = False) -> dict:
        """
        Load and parse JSON artifact
        
        Args:
            path: Either local file path or mlflow://... URI
            force_refresh: Force re-download from MLflow
            
        Returns:
            Parsed JSON as dictionary
        """
        content = self.load_artifact(path, force_refresh)
        return json.loads(content)
    
    def clear_cache(self, max_age_days: int = 30):
        """
        Clear old cached artifacts
        
        Args:
            max_age_days: Remove artifacts older than this many days
        """
        import time
        cutoff = time.time() - (max_age_days * 86400)
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file() and cache_file.stat().st_mtime < cutoff:
                cache_file.unlink()
                removed_count += 1
        
        logger.info(f"Cleared {removed_count} cached artifacts older than {max_age_days} days")
    
    def get_mlflow_uri_example(self) -> str:
        """Return example MLflow URI format"""
        return "mlflow://prompts/summarizer/system_v1"
