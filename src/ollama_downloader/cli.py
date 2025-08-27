import asyncio
import logging
import signal
import sys
from types import FrameType
from typing_extensions import Annotated
import typer
from rich import print as print
from rich import print_json

from environs import env
from ollama_downloader.common import EnvVar
from ollama_downloader.utils import cleanup_unnecessary_files
from ollama_downloader.model_downloader import OllamaModelDownloader
from ollama_downloader.hf_model_downloader import HuggingFaceModelDownloader

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str(EnvVar.LOG_LEVEL, default=EnvVar.DEFAULT__LOG_LEVEL).upper())

# Initialize the Typer application
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)


class OllamaDownloaderCLIApp:
    def __init__(self):
        # Set up signal handlers for graceful shutdown
        self._cleaned = False
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._interrupt_handler)

    def _interrupt_handler(self, signum: int, frame: FrameType | None):
        if frame:
            typer.echo(f"Interrupt signal received at {frame}, performing cleanup...")
        else:
            typer.echo("Interrupt signal received, performing cleanup...")
        self._cleanup()
        typer.echo(f"Cleanup finished. Exiting {OllamaDownloaderCLIApp.__name__}.")
        sys.exit(0)

    def _initialize(self):
        logger.debug(f"Initializing {OllamaDownloaderCLIApp.__name__}...")
        self._model_downloader = OllamaModelDownloader()
        self._hf_model_downloader = HuggingFaceModelDownloader()

    def _cleanup(self):
        if not self._cleaned:
            logger.debug("Running cleanup...")
            if self._model_downloader:
                cleanup_unnecessary_files(self._model_downloader.unnecessary_files)
            if self._hf_model_downloader:
                cleanup_unnecessary_files(self._hf_model_downloader.unnecessary_files)
            self._cleaned = True
        else:
            logger.debug("Cleanup already executed.")

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

    async def _list_models(self):
        return self._model_downloader.update_models_list()

    async def run_list_models(self):
        try:
            self._initialize()
            result = await self._list_models()
            typer.echo(result)
        except Exception as e:
            logger.error(f"Error in listing models. {e}")
        finally:
            self._cleanup()

    async def _list_tags(self, model: str | None, update: bool = False):
        return self._model_downloader.list_models_tags(model=model, update=update)

    async def run_list_tags(self, model: str | None, update: bool = False):
        try:
            self._initialize()
            result = await self._list_tags(model=model, update=update)
            for model_name, tags in result.items():
                print(f"[bold]Model[/bold]: {model_name}")
                if tags:
                    print(f"\t[bold]Tags[/bold] ({len(tags)}): {tags}")
        except Exception as e:
            logger.error(f"Error in listing model tags. {e}")
        finally:
            self._cleanup()

    async def _model_download(self, model_tag: str):
        model, tag = model_tag.split(":") if ":" in model_tag else (model_tag, "latest")
        self._model_downloader.download_model(model=model, tag=tag)

    async def run_model_download(self, model_tag: str):
        try:
            self._initialize()
            await self._model_download(model_tag=model_tag)
        except Exception as e:
            logger.error(f"Error in downloading model. {e}")
        finally:
            self._cleanup()

    async def _hf_model_download(self, user_repo_quant: str):
        self._hf_model_downloader.download_model(user_repo_quant=user_repo_quant)

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
def list_models():
    """Lists all available models in the Ollama library."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_list_models())


@app.command()
def list_tags(
    model: Annotated[
        str | None,
        typer.Argument(
            help="The name of the model to list tags for, e.g., llama3.1. If not provided, tags of all models will be listed."
        ),
    ] = None,
    update: Annotated[
        bool,
        typer.Option(
            help="Force update the model list and its tags before listing.",
        ),
    ] = False,
):
    """Lists all tags for a specific model."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_list_tags(model=model, update=update))


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
