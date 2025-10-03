import logging
import os
from pathlib import Path
from pydantic import AfterValidator, BaseModel, Field, HttpUrl
from typing import Annotated, ClassVar, List, Optional, Tuple

from environs import env

from ollama_downloader.common import EnvVar

logger = logging.getLogger(__name__)


class CustomValidators:
    @staticmethod
    def validate_path_as_dir(path_str: str) -> str:
        p = Path(os.path.expanduser(path_str))
        if not p.exists():
            raise ValueError(f"Path '{path_str}' does not exist.")
        if not p.is_dir():
            raise ValueError(f"Path '{path_str}' is not a valid directory.")
        return path_str

    @staticmethod
    def validate_url(url_str: str) -> str:
        parsed_url = HttpUrl(url_str)
        return str(parsed_url)


class OllamaServer(BaseModel):
    url: Annotated[str, AfterValidator(CustomValidators.validate_url)] = Field(
        default="http://localhost:11434/",
        description="URL of the Ollama server.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the Ollama server, if required.",
    )
    remove_downloaded_on_error: bool = Field(
        default=True,
        description="Whether to remove downloaded files if the downloaded model cannot be found on the Ollama server, or the Ollama server cannot be accessed.",
    )


class OllamaLibrary(BaseModel):
    models_path: Annotated[
        str, AfterValidator(CustomValidators.validate_path_as_dir)
    ] = Field(
        default="~/.ollama/models",
        description="Path to the Ollama models on the filesystem. This should be a directory where model BLOBs and manifest metadata are stored.",
    )
    registry_base_url: Annotated[str, AfterValidator(CustomValidators.validate_url)] = (
        Field(
            default="https://registry.ollama.ai/v2/library/",
            description="URL of the remote registry for Ollama models.",
        )
    )
    library_base_url: Annotated[str, AfterValidator(CustomValidators.validate_url)] = (
        Field(
            default="https://ollama.com/library/",
            description="Base URL for the Ollama library. This is used to web scrape model metadata.",
        )
    )
    verify_ssl: Optional[bool] = Field(
        default=True,
        description="Whether to verify SSL certificates when connecting to the Ollama server or registry. Set to False to disable SSL verification (not recommended for production use).",
    )
    timeout: Optional[float] = Field(
        default=120.0,
        description="Timeout for HTTP requests to the Ollama server or registry, in seconds.",
    )
    user_group: Optional[Tuple[str, str]] = Field(
        default=None,
        description="A tuple specifying the username and the group that should own the Ollama models path. If not provided, the current user and group will be used.",
    )


class AppSettings(BaseModel):
    ollama_server: OllamaServer = Field(
        default=OllamaServer(),
        description="Settings for the Ollama server connection.",
    )
    ollama_library: OllamaLibrary = Field(
        default=OllamaLibrary(),
        description="Settings for accessing the Ollama library and storing locally.",
    )

    _instance: ClassVar = None

    def __new__(cls: type["AppSettings"]) -> "AppSettings":
        if cls._instance is None:
            # Create instance using super().__new__ to bypass any recursion
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance

    @staticmethod
    def load_or_create_default(
        settings_file: str = EnvVar.DEFAULT__OD_SETTINGS_FILE,
    ) -> "AppSettings | None":
        """
        Load settings from the configuration file, or create default settings if the file does not exist.

        Returns:
            AppSettings: The application settings loaded from the configuration file,
            or default settings if the file does not exist.
        """
        settings = AppSettings.load_settings(settings_file)
        if settings is None:
            # This will be a singleton instance
            settings = AppSettings()
            if not AppSettings.save_settings(settings, settings_file):
                return None
        return settings

    @staticmethod
    def load_settings(
        settings_file: str = env.str(
            EnvVar.OD_SETTINGS_FILE, default=EnvVar.DEFAULT__OD_SETTINGS_FILE
        ),
    ) -> "AppSettings | None":
        """
        Load settings from the configuration file.

        Returns:
            AppSettings: The application settings loaded from the configuration file.
            If the file does not exist or cannot be parsed, returns None.
        """
        try:
            with open(settings_file, "r") as f:
                # Parse the JSON file into the AppSettings model
                return_value = AppSettings.model_validate_json(f.read())
            return return_value
        except FileNotFoundError:
            logger.error(f"Configuration file {settings_file} not found.")
        except Exception as e:
            logger.error(f"Error loading settings from {settings_file}. {e}")
        return None

    @staticmethod
    def save_settings(
        settings: "AppSettings",
        settings_file: str = env.str(
            EnvVar.OD_SETTINGS_FILE, default=EnvVar.DEFAULT__OD_SETTINGS_FILE
        ),
    ) -> bool:
        """
        Save the application settings to the configuration file.

        Returns:
            bool: True if settings were saved successfully, False otherwise.
        """
        try:
            config_dir = os.path.dirname(settings_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=False)
            with open(settings_file, "w") as f:
                f.write(settings.model_dump_json(indent=4))
            logger.info(f"Settings saved to {settings_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings to {settings_file}. {e}")
            return False


class ImageManifestConfig(BaseModel):
    mediaType: str = Field(
        ...,
        description="The media type of the image manifest configuration.",
    )
    size: int = Field(
        ...,
        description="The size of the image manifest configuration in bytes.",
    )
    digest: str = Field(
        ...,
        description="The digest of the image manifest configuration, used for content addressing.",
    )


class ImageManifestLayerEntry(BaseModel):
    mediaType: str = Field(
        ...,
        description="The media type of the layer.",
    )
    size: int = Field(
        ...,
        description="The size of the layer in bytes.",
    )
    digest: str = Field(
        ...,
        description="The digest of the layer, used for content addressing.",
    )
    # Note that these URLs may not be present in all manifests and may not be possible to validate as HttpUrls.
    urls: Optional[list[str]] = Field(
        default=None,
        description="Optional list of URLs where the layer can be downloaded from. This is useful for layers that are hosted on multiple locations.",
    )


class ImageManifest(BaseModel):
    # See: https://distribution.github.io/distribution/spec/manifest-v2-2/#image-manifest
    schemaVersion: int = Field(
        ...,
        description="The schema version of the image manifest.",
    )
    mediaType: str = Field(
        ...,
        description="The media type of the image manifest.",
    )
    config: ImageManifestConfig = Field(
        ...,
        description="Configuration for the image manifest, including media type, size, and digest.",
    )
    layers: Optional[List[ImageManifestLayerEntry]] = Field(
        None,
        description="List of layers in the image manifest, each with its media type, size, and digest.",
    )
