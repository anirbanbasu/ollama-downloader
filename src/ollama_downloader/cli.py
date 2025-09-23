import asyncio
import grp
import logging
import os
import platform
import pwd
import re
import signal
import sys
from types import FrameType
from typing import Any, Dict, Optional
from typing_extensions import Annotated
import typer
from rich import print as print
from rich import print_json
import psutil

from ollama_downloader.common import OllamaSystemInfo
from ollama_downloader.data.data_models import AppSettings
from ollama_downloader.downloader.ollama_model_downloader import OllamaModelDownloader
from ollama_downloader.downloader.hf_model_downloader import HuggingFaceModelDownloader

from ollama import Client as OllamaClient

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
        self._model_downloader: OllamaModelDownloader = None
        self._hf_model_downloader: HuggingFaceModelDownloader = None

    def _interrupt_handler(self, signum: int, frame: FrameType | None):
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

    async def _show_config(self):
        return self._model_downloader.settings.model_dump_json()

    async def run_show_config(self):
        try:
            self._initialize()
            result = await self._show_config()
            print_json(json=result)
        except Exception as e:
            logger.error(f"Error in showing config. {e}")
        finally:
            self._cleanup()

    async def _auto_config(self):
        logger.warning(
            "Automatic configuration is an experimental feature. Its output maybe incorrect!"
        )
        system_info = OllamaSystemInfo()
        if system_info.is_windows():
            raise NotImplementedError(
                "Automatic configuration is not supported on Windows yet."
            )
        super_user_needed = False
        super_user_needed = super_user_needed or system_info.infer_listening_on() in [
            None,
            "",
        ]
        super_user_needed = (
            super_user_needed or system_info.infer_models_dir_path() in [None, ""]
        )
        if super_user_needed:
            return {}
        inferred_settings = AppSettings()
        inferred_settings.ollama_server.url = system_info.listening_on
        inferred_settings.ollama_library.models_path = system_info.models_dir_path
        if system_info.is_likely_daemon():
            if system_info.is_macos():
                logger.warning(
                    "Automatic configuration on macOS maybe flawed if Ollama is configured to run as a background service."
                )
            inferred_settings.ollama_library.user_group = (
                system_info.process_owner[0],
                system_info.process_owner[2],
            )
        return inferred_settings.model_dump_json()

    async def old_auto_config(self):
        logger.warning(
            "Automatic configuration is a future experimental idea, which has not been implemented yet. Stay tuned!"
        )
        logger.warning(
            "Some relevant information about Ollama will be displayed but these are based on assumptions that may not always be true."
        )
        # TODO: The following stuff must go in its own module/class with its own tests. This is just a placeholder.
        # The idea is to gather relevant information about the Ollama installation and use it to infer a configuration.
        # 1. Check if Ollama is installed and running.
        # 2. Output the IP/interface and port the Ollama server is listening on.
        # 3. Output the Ollama executable path.
        # 4. Output user/group Ollama is running as.
        # 5. Attempt to connect to the running Ollama server and try to infer the model storage path from the first modelfile.

        if platform.system() == "Windows":
            # Don't bother going there until we have a real Windows machine to test on.
            raise NotImplementedError(
                "Automatic configuration is not supported on Windows yet. Probably never will be!"
            )
        relevant_info: Dict[str, Any] = {}

        pid: int = -1
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == "ollama":
                pid = proc.info["pid"]
        if pid < 0:
            raise RuntimeError(
                "Ollama process not found. Is Ollama installed and running?"
            )
        process = psutil.Process(pid)
        relevant_info["process"] = {"name": process.name(), "pid": pid}
        parent_id = process.ppid()
        relevant_info["parent"] = {
            "name": psutil.Process(parent_id).name(),
            "pid": parent_id,
        }
        # User and Group Information
        username = process.username()

        owner = {}
        effective_uid = process.uids().effective
        effective_gid = process.gids().effective
        owner["user"] = {"name": username, "uid": effective_uid}
        owner["group"] = {
            "name": grp.getgrgid(effective_gid).gr_name,
            "gid": effective_gid,
        }
        relevant_info["owner"] = owner

        relevant_info["sudo_needed_to_download_models"] = os.getuid() != effective_uid

        relevant_info["executable"] = process.exe()
        relevant_info["cmdline"] = process.cmdline()

        explicit_models_dir = process.environ().get("OLLAMA_MODELS", None)
        relevant_info["model_dir_explicitly_specified"] = (
            explicit_models_dir is not None
        )

        relevant_info["http_proxy_specified"] = (
            process.environ().get("HTTP_PROXY", None) is not None
            or process.environ().get("http_proxy", None) is not None
            or process.environ().get("HTTPS_PROXY", None) is not None
            or process.environ().get("https_proxy", None) is not None
        )

        listening_on = []
        ollama_url = ""
        for conn in process.net_connections(kind="inet"):
            if conn.status == psutil.CONN_LISTEN:
                laddr = (
                    f"{conn.laddr.ip}:{conn.laddr.port}"
                    if conn.laddr.ip != "::"
                    else f"127.0.0.1:{conn.laddr.port}"
                )
                listening_on.append(laddr)
                ollama_url = f"http://{laddr}"
                # Just take the first listening address
                break
        relevant_info["listening_on"] = listening_on

        # FIXME: This is not a good idea, since we are including a specific check
        # but most systems use this init system so it maybe fine.
        systemd_service = "/etc/systemd/system/ollama.service"
        systemd_daemon = os.path.exists(systemd_service)

        ollama_client = OllamaClient(host=ollama_url)
        models_list = ollama_client.list().models
        if explicit_models_dir is not None:
            relevant_info["likely_models_path"] = explicit_models_dir
        elif systemd_daemon:
            # https://ollama.com/install.sh :: function configure_systemd()
            # if systemd is the init system we use the /usr/share/ollama directory for all installs
            relevant_info["likely_models_path"] = "/usr/share/ollama/.ollama/models"
        elif len(models_list) > 0:
            modelfile_text = ollama_client.show(models_list[0].model).modelfile
            pattern = re.compile(
                r"^\s*FROM\s+(.+?)(?:\s*#.*)?$", re.IGNORECASE | re.MULTILINE
            )
            match = pattern.search(modelfile_text)
            if match:
                model_blob_path = match.group(1).strip()
                likely_models_dir = str(
                    os.path.dirname(os.path.dirname(model_blob_path))
                )
                relevant_info["likely_models_path"] = likely_models_dir.replace(
                    os.path.expanduser("~"), "~"
                )
                real_models_path = os.path.realpath(likely_models_dir)
                is_symlink_in_path = real_models_path != likely_models_dir
                if is_symlink_in_path:
                    relevant_info["symlink_in_models_path"] = is_symlink_in_path
                    relevant_info["likely_real_models_path"] = real_models_path
                    stat_info = os.stat(real_models_path)
                    relevant_info["likely_real_models_path_owner"] = {
                        "user": {
                            "name": pwd.getpwuid(stat_info.st_uid).pw_name,
                            "uid": stat_info.st_uid,
                        },
                        "group": {
                            "name": grp.getgrgid(stat_info.st_gid).gr_name,
                            "gid": stat_info.st_gid,
                        },
                    }
        else:
            # We could select ~/.ollama/models as a default directory
            relevant_info["likely_models_path"] = "~/.ollama/models"
            logger.warning(
                "No models found in the running Ollama instance. Models path cannot be computed. Using defaults."
            )

        open_files = []
        for file in process.open_files():
            open_files.append(file.path)
        relevant_info["ollama_open_files"] = open_files

        # auto-config does not need to fail if we can't detect if ollama is a daemon or not
        try:
            relevant_info["ollama_is_likely_daemon"] = systemd_daemon or (
                process.terminal() is None
                and process.status()
                not in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
                and process.ppid() != 1
            )
        except Exception as e:
            if isinstance(e, psutil.AccessDenied):
                logger.info(
                    "Seems like you need to run this command with super-user permissions. Try `sudo`!"
                )

        current_settings = self._model_downloader.settings

        new_settings = current_settings.model_dump()
        new_settings["ollama_library"]["user_group"] = (
            owner["user"]["name"],
            owner["group"]["name"],
        )
        new_settings["ollama_library"]["models_path"] = relevant_info[
            "likely_models_path"
        ]

        for entry in new_settings["ollama_library"]:
            value = new_settings["ollama_library"][entry]
            current_value = current_settings.ollama_library.__dict__[entry]
            if current_value != value:
                logger.info(
                    f"Change value of {entry} from {current_value} to {value}. Updating the `conf/settings.json` may be nessesary."
                )

        # we do not update anything without the user consent
        return current_settings.model_dump_json()

    async def run_auto_config(self):
        try:
            # self._initialize()
            result = await self._auto_config()
            if result != {}:
                print_json(json=result)
        except Exception as e:
            logger.error(f"Error in generating automatic config. {e}")
            if isinstance(e, psutil.AccessDenied):
                logger.info(
                    "Seems like you need to run this command with super-user permissions. Try `sudo`!"
                )
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
def auto_config():
    """Displays an automatically inferred configuration."""
    app_handler = OllamaDownloaderCLIApp()
    asyncio.run(app_handler.run_auto_config())


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
