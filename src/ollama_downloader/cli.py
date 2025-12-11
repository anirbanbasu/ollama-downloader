import asyncio
import logging
import signal
import sys
from importlib.metadata import version as metadata_version
from types import FrameType
from typing import Annotated

import psutil
import typer
from httpx import HTTPStatusError
from rich import print as print, print_json

from ollama_downloader.data.data_models import AppSettings
from ollama_downloader.downloader.hf_model_downloader import HuggingFaceModelDownloader
from ollama_downloader.downloader.ollama_model_downloader import OllamaModelDownloader
from ollama_downloader.sysinfo import OllamaSystemInfo

# Initialize the logger
logger = logging.getLogger(__name__)

# Initialize the Typer application
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)


class OllamaDownloaderCLIApp:
    """Class to handle the CLI application logic for Ollama Downloader."""

    def __init__(self):
        # Set up signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._interrupt_handler)
        self._model_downloader: OllamaModelDownloader = None
        self._hf_model_downloader: HuggingFaceModelDownloader = None

    def _interrupt_handler(self, signum: int, frame: FrameType | None):  # pragma: no cover
        logger.warning("Interrupt signal received, performing clean shutdown")
        logger.debug(f"Interrupt signal number: {signum}. Frame: {frame}")
        # Cleanup will be performed due to the finally block in each command
        sys.exit(0)

    def _initialize(self):
        logger.debug("Initializing downloaders...")
        if not self._model_downloader:
            self._model_downloader = OllamaModelDownloader()
        if not self._hf_model_downloader:
            self._hf_model_downloader = HuggingFaceModelDownloader()

    def _cleanup(self):
        logger.debug("Running cleanup...")

        if self._model_downloader:
            self._model_downloader.cleanup_unnecessary_files()
        if self._hf_model_downloader:
            self._hf_model_downloader.cleanup_unnecessary_files()

        logger.debug("Cleanup completed.")

    async def _version(self):
        package_name = "ollama-downloader"
        name_splits = package_name.split("-")
        abbreviation = f"{name_splits[0][0]}{name_splits[1][0]}"
        return f"{package_name} ({abbreviation}) version {metadata_version(package_name)}"

    async def run_version(self):
        """Run the version command and print the version information."""
        try:
            result = await self._version()
            print(result)
        except Exception as e:  # pragma: no cover
            # This will never happen but it is here for completeness
            logger.error(f"Error in getting version. {e}")

    async def _show_config(self):
        return self._model_downloader.settings.model_dump_json()

    async def run_show_config(self):
        """Run the show_config command and print the configuration as JSON."""
        try:
            self._initialize()
            result = await self._show_config()
            print_json(json=result)
        except Exception as e:  # pragma: no cover
            logger.error(f"Error in showing config. {e}")
        finally:
            self._cleanup()

    async def _auto_config(self):
        logger.warning("Automatic configuration is an experimental feature. Its output maybe incorrect!")
        system_info = OllamaSystemInfo()
        if system_info.is_windows():
            raise NotImplementedError("Automatic configuration is not supported on Windows yet.")
        super_user_maybe_needed = False
        super_user_maybe_needed = super_user_maybe_needed or system_info.infer_listening_on() in [
            None,
            "",
        ]
        super_user_maybe_needed = super_user_maybe_needed or system_info.infer_models_dir_path() in [
            None,
            "",
        ]
        if super_user_maybe_needed:  # pragma: no cover
            raise RuntimeError(
                "Automatic configuration could not infer some settings. Maybe super-user permissions are necessary. Or, perhaps, Ollama has no models installed yet."
            )
        else:  # pragma: no cover
            inferred_settings = AppSettings()
            inferred_settings.ollama_server.url = system_info.listening_on
            inferred_settings.ollama_library.models_path = system_info.models_dir_path
            if system_info.is_likely_daemon():  # pragma: no cover
                if system_info.is_macos():
                    logger.warning(
                        "Automatic configuration on macOS maybe flawed if Ollama is configured to run as a system background service."
                    )
                inferred_settings.ollama_library.user_group = (
                    system_info.process_owner[0],
                    system_info.process_owner[2],
                )
            return inferred_settings.model_dump_json()

    async def run_auto_config(self):
        """Run the auto_config command and print the automatically inferred configuration."""
        try:
            result = await self._auto_config()
            if result != {}:  # pragma: no cover
                print_json(json=result)
        except Exception as e:  # pragma: no cover
            logger.error(f"Error in generating automatic config. {e}")
            if isinstance(e, psutil.AccessDenied):
                logger.info("Seems like you need to run this command with super-user permissions. Try `sudo`!")
        finally:
            self._cleanup()

    async def _list_models(self, page: int | None = None, page_size: int | None = None):
        return self._model_downloader.list_available_models(page=page, page_size=page_size)

    async def run_list_models(self, page: int | None = None, page_size: int | None = None):
        """Lists all available models in the Ollama library. If pagination options are not provided, all models will be listed."""
        try:
            self._initialize()
            result = await self._list_models(page=page, page_size=page_size)
            if page and page_size and page_size >= len(result):
                print(f"Model identifiers: ({len(result)}, page {page}): {result}")
            else:
                print(f"Model identifiers: ({len(result)}): {result}")
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Error in listing models. {e}{'\n' + e.response.text if isinstance(e, HTTPStatusError) else ''}"
            )
        finally:
            self._cleanup()

    async def _list_tags(self, model_identifier: str):
        return self._model_downloader.list_model_tags(model_identifier=model_identifier)

    async def run_list_tags(self, model_identifier: str):
        """Lists all tags for a specific model."""
        try:
            self._initialize()
            result = await self._list_tags(model_identifier=model_identifier)
            print(f"Model tags: ({len(result)}): {result}")
        except Exception as e:
            logger.error(
                f"Error in listing model tags. {e}{'\n' + e.response.text if isinstance(e, HTTPStatusError) else ''}"
            )
        finally:
            self._cleanup()

    async def _model_download(self, model_tag: str):
        self._model_downloader.download_model(model_tag)

    async def run_model_download(self, model_tag: str):
        """Downloads a specific Ollama model with the given tag."""
        try:
            self._initialize()
            await self._model_download(model_tag=model_tag)
        except Exception as e:
            logger.error(
                f"Error in downloading model. {e}{'\n' + e.response.text if isinstance(e, HTTPStatusError) else ''}"
            )
        finally:
            self._cleanup()

    async def _hf_list_models(self, page: int | None = None, page_size: int | None = None):
        return self._hf_model_downloader.list_available_models(page=page, page_size=page_size)

    async def run_hf_list_models(self, page: int | None = None, page_size: int | None = None):
        """Lists available models from Hugging Face that can be downloaded into Ollama."""
        try:
            self._initialize()
            result = await self._hf_list_models(page=page, page_size=page_size)
            if page:
                print(f"Model identifiers: ({len(result)}, page {page}): {result}")
            else:  # This won't really happen as we always pass a value of page by default.
                print(f"Model identifiers: ({len(result)}): {result}")  # pragma: no cover
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Error in listing models. {e}{'\n' + e.response.text if isinstance(e, HTTPStatusError) else ''}"
            )
        finally:
            self._cleanup()

    async def _hf_list_tags(self, model_identifier: str):
        return self._hf_model_downloader.list_model_tags(model_identifier=model_identifier)

    async def run_hf_list_tags(self, model_identifier: str):
        """Lists all available quantisations as tags for a Hugging Face model that can be downloaded into Ollama."""
        try:
            self._initialize()
            result = await self._hf_list_tags(model_identifier=model_identifier)
            print(f"Model tags: ({len(result)}): {result}")
        except Exception as e:
            logger.error(
                f"Error in listing model tags. {e}{'\n' + e.response.text if isinstance(e, HTTPStatusError) else ''}"
            )
        finally:
            self._cleanup()

    async def _hf_model_download(self, user_repo_quant: str):
        self._hf_model_downloader.download_model(model_identifier=user_repo_quant)

    async def run_hf_model_download(self, user_repo_quant: str):
        """Downloads a specified Hugging Face model."""
        try:
            self._initialize()
            await self._hf_model_download(user_repo_quant=user_repo_quant)
        except Exception as e:
            logger.error(
                f"Error in downloading Hugging Face model. {e}{'\n' + e.response.text if isinstance(e, HTTPStatusError) else ''}"
            )
        finally:
            self._cleanup()


@app.command()
def version():
    """Shows the app version of Ollama downloader."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_version())


@app.command()
def show_config():
    """Shows the application configuration as JSON."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_show_config())


@app.command()
def auto_config():
    """Displays an automatically inferred configuration."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_auto_config())


@app.command()
def list_models(
    page: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="The page number to retrieve (1-indexed).",
        ),
    ] = None,
    page_size: Annotated[
        int | None,
        typer.Option(
            min=1,
            max=100,
            help="The number of models to retrieve per page.",
        ),
    ] = None,
):
    """Lists all available models in the Ollama library.

    If pagination options are not provided, all models will be listed.
    """
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_list_models(page=page, page_size=page_size))


@app.command()
def list_tags(
    model_identifier: Annotated[
        str,
        typer.Argument(help="The name of the model to list tags for, e.g., llama3.1."),
    ],
):
    """Lists all tags for a specific model."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_list_tags(model_identifier=model_identifier))


@app.command()
def model_download(
    model_tag: Annotated[
        str,
        typer.Argument(
            help="The name of the model and a specific to download, specified as <model>:<tag>, e.g., llama3.1:8b. If no tag is specified, 'latest' will be assumed.",
        ),
    ],
):
    """Downloads a specific Ollama model with the given tag."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_model_download(model_tag=model_tag))


@app.command()
def hf_list_models(
    page: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="The page number to retrieve (1-indexed).",
        ),
    ] = 1,
    page_size: Annotated[
        int | None,
        typer.Option(
            min=1,
            max=100,
            help="The number of models to retrieve per page.",
        ),
    ] = 25,
):
    """Lists available models from Hugging Face that can be downloaded into Ollama."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_hf_list_models(page, page_size))


@app.command()
def hf_list_tags(
    model_identifier: Annotated[
        str,
        typer.Argument(help="The name of the model to list tags for, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF."),
    ],
):
    """Lists all available quantisations as tags for a Hugging Face model that can be downloaded into Ollama.

    Note that these are NOT the same as Hugging Face model tags.
    """
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_hf_list_tags(model_identifier=model_identifier))


@app.command()
def hf_model_download(
    user_repo_quant: Annotated[
        str,
        typer.Argument(
            help="The name of the specific Hugging Face model to download, specified as <username>/<repository>:<quantisation>, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M.",
        ),
    ],
):
    """Downloads a specified Hugging Face model."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_hf_model_download(user_repo_quant=user_repo_quant))


def main():
    """Main entry point for the CLI application."""
    # Run the Typer app
    app()  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    main()
