import os
from pydantic import BaseModel, Field, DirectoryPath
from typing import ClassVar, Optional, Tuple
from pathlib import Path

from ollama_downloader.common import logger


class OllamaServer(BaseModel):
    url: str = Field(
        default="http://localhost:11434",
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


class OllamaLibrary(BaseModel, validate_assignment=True):
    models_path: DirectoryPath = Field(
        default=Path("~/.ollama/models"),
        description="Path to the Ollama models on the filesystem. This should be a directory where model BLOBs and manifest metadata are stored.",
    )
    models_tags_cache: str = Field(
        default="models_tags.json",
        description="Path to the cache file for model tags. This file is used to store the tags of models to avoid repeated requests to the Ollama server.",
    )
    registry_base_url: Optional[str] = Field(
        default="https://registry.ollama.ai/v2/library/",
        description="URL of the remote registry for Ollama models. If not provided, local storage will be used.",
    )
    library_base_url: Optional[str] = Field(
        default="https://ollama.com/library",
        description="Base URL for the Ollama library. This is used to web scrape model metadata.",
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
    # ClassVar annotation is necessary to differentiate these from Pydantic fields
    _settings_dir: ClassVar[str] = "conf"
    _settings_file: ClassVar[str] = os.path.join(_settings_dir, "settings.json")

    ollama_server: OllamaServer = Field(
        default=OllamaServer(),
        description="Settings for the Ollama server connection.",
    )
    ollama_library: OllamaLibrary = Field(
        default=OllamaLibrary(),
        description="Settings for accessing the Ollama library and storing locally.",
    )

    def read_settings(self) -> bool:
        """
        Load settings from the configuration file.

        Returns:
            bool: True if settings were loaded successfully, False otherwise.
        """
        try:
            with open(AppSettings._settings_file, "r") as f:
                # Parse the JSON file into the AppSettings model
                new_instance = AppSettings.model_validate_json(f.read())
            # Update the current instance with the loaded settings
            self.__dict__.update(new_instance.__dict__)
            return True
        except FileNotFoundError:
            logger.error(
                f"[bold red]Configuration file {AppSettings._settings_file} not found.[/bold red]"
            )
            return False
        except Exception as e:
            logger.exception(
                f"[bold red]Error loading settings from {AppSettings._settings_file}: {e}[/bold red]"
            )
            return False

    def save_settings(self) -> bool:
        """
        Save the application settings to the configuration file.

        Returns:
            bool: True if settings were saved successfully, False otherwise.
        """
        try:
            os.makedirs(AppSettings._settings_dir, exist_ok=True)
            with open(AppSettings._settings_file, "w") as f:
                f.write(self.model_dump_json(indent=4))
            logger.info(
                f"[bold green]Settings saved to {AppSettings._settings_file}[/bold green]"
            )
            return True
        except Exception as e:
            logger.exception(
                f"[bold red]Error saving settings to {AppSettings._settings_file}: {e}[/bold red]"
            )
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
    layers: list[ImageManifestLayerEntry] = Field(
        ...,
        description="List of layers in the image manifest, each with its media type, size, and digest.",
    )
