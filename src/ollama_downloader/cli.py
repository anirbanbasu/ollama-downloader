import asyncio
import logging
import signal
import sys
from types import FrameType
from typing import Optional
from typing_extensions import Annotated
import typer
from rich import print as print
from rich import print_json

from ollama_downloader.downloader.ollama_model_downloader import OllamaModelDownloader
from ollama_downloader.downloader.hf_model_downloader import HuggingFaceModelDownloader

# Initialize the logger
logger = logging.getLogger(__name__)

# Initialize the Typer application
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)


class OllamaDownloaderCLIApp:
    def __init__(self):
        # Set up signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._interrupt_handler)

    def _interrupt_handler(self, signum: int, frame: FrameType | None):
        logger.warning("Interrupt signal received, performing clean shutdown")
        logger.debug(f"Interrupt signal number: {signum}. Frame: {frame}")
        # Cleanup will be performed due to the finally block in each command
        sys.exit(0)

    def _initialize(self):
        logger.debug("Initializing downloaders...")
        self._model_downloader = OllamaModelDownloader()
        self._hf_model_downloader = HuggingFaceModelDownloader()

    def _cleanup(self):
        logger.debug("Running cleanup...")

        if self._model_downloader:
            self._model_downloader.cleanup_unnecessary_files()
        if self._hf_model_downloader:
            self._hf_model_downloader.cleanup_unnecessary_files()

        logger.debug("Cleanup completed.")

    async def _show_config(self):
        return self._model_downloader.settings.model_dump_json()

    async def run_show_config(self):
        try:
            self._initialize()
            result = await self._show_config()
            # TODO: Should we pretty-print the JSON with indentation here or with the Pydantic's `model_dump_json`?
            print_json(json=result)
        except Exception as e:
            logger.error(f"Error in showing config. {e}")
        finally:
            self._cleanup()

    async def _list_models(self, page: int | None = None, page_size: int | None = None):
        return self._model_downloader.list_available_models(
            page=page, page_size=page_size
        )

    async def run_list_models(
        self, page: int | None = None, page_size: int | None = None
    ):
        try:
            self._initialize()
            result = await self._list_models()
            filtered_result = result
            if page_size and page:
                # Adjust page number for 0-based index
                start_index = (page - 1) * page_size
                end_index = start_index + page_size
                filtered_result = result[start_index:end_index]
            if len(filtered_result) == 0:
                logger.warning(
                    f"No models found for the specified page {page} and page size {page_size}. Showing all models instead."
                )
                filtered_result = result
                page = None
            if page:
                print(
                    f"Model identifiers: ({len(filtered_result)}, page {page}): {filtered_result}"
                )
            else:
                print(f"Model identifiers: ({len(filtered_result)}): {filtered_result}")
        except Exception as e:
            logger.error(f"Error in listing models. {e}")
        finally:
            self._cleanup()

    async def _list_tags(self, model_identifier: str):
        return self._model_downloader.list_model_tags(model_identifier=model_identifier)

    async def run_list_tags(self, model_identifier: str):
        try:
            self._initialize()
            result = await self._list_tags(model_identifier=model_identifier)
            print(f"Model tags: ({len(result)}): {result}")
        except Exception as e:
            logger.error(f"Error in listing model tags. {e}")
        finally:
            self._cleanup()

    async def _model_download(self, model_tag: str):
        self._model_downloader.download_model(model_tag)

    async def run_model_download(self, model_tag: str):
        try:
            self._initialize()
            await self._model_download(model_tag=model_tag)
        except Exception as e:
            logger.error(f"Error in downloading model. {e}")
        finally:
            self._cleanup()

    async def _hf_list_models(
        self, page: int | None = None, page_size: int | None = None
    ):
        return self._hf_model_downloader.list_available_models(
            page=page, page_size=page_size
        )

    async def run_hf_list_models(
        self, page: int | None = None, page_size: int | None = None
    ):
        try:
            self._initialize()
            result = await self._hf_list_models(page=page, page_size=page_size)
            if page:
                print(f"Model identifiers: ({len(result)}, page {page}): {result}")
            else:
                print(f"Model identifiers: ({len(result)}): {result}")
        except Exception as e:
            logger.error(f"Error in listing models. {e}")
        finally:
            self._cleanup()

    async def _hf_list_tags(self, model_identifier: str):
        return self._hf_model_downloader.list_model_tags(
            model_identifier=model_identifier
        )

    async def run_hf_list_tags(self, model_identifier: str):
        try:
            self._initialize()
            result = await self._hf_list_tags(model_identifier=model_identifier)
            print(f"Model tags: ({len(result)}): {result}")
        except Exception as e:
            logger.error(f"Error in listing model tags. {e}")
        finally:
            self._cleanup()

    async def _hf_model_download(self, user_repo_quant: str):
        self._hf_model_downloader.download_model(model_identifier=user_repo_quant)

    async def run_hf_model_download(self, user_repo_quant: str):
        try:
            self._initialize()
            await self._hf_model_download(user_repo_quant=user_repo_quant)
        except Exception as e:
            logger.error(f"Error in downloading Hugging Face model. {e}")
        finally:
            self._cleanup()


@app.command()
def show_config():
    """Shows the application configuration as JSON."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_show_config())


@app.command()
def list_models(
    page: Annotated[
        Optional[int],
        typer.Option(
            min=1,
            help="The page number to retrieve (1-indexed).",
        ),
    ] = None,
    page_size: Annotated[
        Optional[int],
        typer.Option(
            min=1,
            max=100,
            help="The number of models to retrieve per page.",
        ),
    ] = None,
):
    """Lists all available models in the Ollama library. If pagination options are not provided, all models will be listed."""
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
        Optional[int],
        typer.Option(
            min=1,
            help="The page number to retrieve (1-indexed).",
        ),
    ] = 1,
    page_size: Annotated[
        Optional[int],
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
        typer.Argument(
            help="The name of the model to list tags for, e.g., bartowski/Llama-3.2-1B-Instruct-GGUF."
        ),
    ],
):
    """
    Lists all available quantisations as tags for a Hugging Face model that can be downloaded into Ollama.
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
    app()


if __name__ == "__main__":
    main()
