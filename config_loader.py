"""
Configuration Loader - Load agent configs with support for MLflow and local artifacts
"""

import os
import yaml
import json
import logging
from typing import Optional, Dict, Any
from artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Centralized configuration loader with artifact support"""
    
    def __init__(self, artifact_manager: Optional[ArtifactManager] = None):
        """
        Initialize ConfigLoader
        
        Args:
            artifact_manager: ArtifactManager instance (created if not provided)
        """
        self.artifact_manager = artifact_manager or ArtifactManager()
    
    def _configure_mlflow(self, config: Dict[str, Any]):
        """Apply MLflow settings from config if provided."""
        mlflow_config = config.get("mlflow")
        if not isinstance(mlflow_config, dict):
            return
        
        tracking_uri = mlflow_config.get("tracking_uri")
        registry_uri = mlflow_config.get("registry_uri")
        default_run_id = mlflow_config.get("run_id")
        run_id_map = mlflow_config.get("run_id_map")
        artifact_uri_map = mlflow_config.get("artifact_uri_map")
        fallback_paths = mlflow_config.get("fallback_paths")

        if not isinstance(run_id_map, dict):
            run_id_map = None
        if not isinstance(artifact_uri_map, dict):
            artifact_uri_map = None
        if not isinstance(fallback_paths, dict):
            fallback_paths = None
        
        self.artifact_manager.configure(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            default_run_id=default_run_id,
            run_id_map=run_id_map,
            artifact_uri_map=artifact_uri_map,
            fallback_paths=fallback_paths,
        )
    
    def load_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Load agent configuration from YAML file
        
        Args:
            agent_name: Name of the agent (matches config filename without .yaml)
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file not found
        """
        config_path = f"configs/{agent_name}.yaml"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config found for agent '{agent_name}' at {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if isinstance(config, dict):
            self._configure_mlflow(config)
        
        logger.debug(f"Loaded config for agent '{agent_name}' from {config_path}")
        return config
    
    def resolve_artifact_path(self, path: str, force_refresh: bool = False) -> str:
        """
        Resolve artifact path - load from MLflow if mlflow:// URI, otherwise return local path
        
        Args:
            path: Either local file path or mlflow://... URI
            force_refresh: Force re-download from MLflow
            
        Returns:
            Artifact content as string
            
        Raises:
            FileNotFoundError: If artifact not found
        """
        return self.artifact_manager.load_artifact(path, force_refresh)
    
    def resolve_json_artifact(self, path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Resolve and parse JSON artifact
        
        Args:
            path: Either local file path or mlflow://... URI
            force_refresh: Force re-download from MLflow
            
        Returns:
            Parsed JSON as dictionary
        """
        return self.artifact_manager.load_json_artifact(path, force_refresh)
    
    def load_prompt(self, path: str) -> str:
        """
        Load prompt text from file or MLflow
        
        Args:
            path: Either local file path or mlflow://... URI
            
        Returns:
            Prompt content as string
        """
        try:
            content = self.resolve_artifact_path(path)
            logger.debug(f"Loaded prompt from {path}")
            return content
        except FileNotFoundError as e:
            logger.error(f"Failed to load prompt from {path}: {e}")
            raise
    
    def load_schema(self, path: str) -> Dict[str, Any]:
        """
        Load JSON schema from file or MLflow
        
        Args:
            path: Either local file path or mlflow://... URI
            
        Returns:
            Parsed schema as dictionary
        """
        try:
            schema = self.resolve_json_artifact(path)
            logger.debug(f"Loaded schema from {path}")
            return schema
        except FileNotFoundError as e:
            logger.error(f"Failed to load schema from {path}: {e}")
            raise


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get or create global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def set_config_loader(loader: ConfigLoader):
    """Set global config loader instance"""
    global _config_loader
    _config_loader = loader


def load_config(agent_name: str) -> Dict[str, Any]:
    """Load agent configuration (convenience function)"""
    return get_config_loader().load_config(agent_name)


def load_prompt(path: str) -> str:
    """Load prompt text (convenience function)"""
    return get_config_loader().load_prompt(path)


def load_schema(path: str) -> Dict[str, Any]:
    """Load JSON schema (convenience function)"""
    return get_config_loader().load_schema(path)
