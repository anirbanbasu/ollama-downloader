import logging
import os
import ssl
from typing import Set

import certifi
import httpx
import platform

from environs import env
from ollama_downloader.common import EnvVar
from ollama_downloader.data.data_models import AppSettings

logger = logging.getLogger(__name__)

user_agent = f"{env.str(EnvVar.OD_UA_NAME_VER, default=EnvVar.DEFAULT__OD_UA_NAME_VER)} ({platform.platform()} {platform.system()}-{platform.release()} Python-{platform.python_version()})"

CONF_DIR = env.str(EnvVar.OD_CONF_DIR, default=EnvVar.DEFAULT__OD_CONF_DIR)
SETTINGS_FILE = os.path.join(
    CONF_DIR, env.str(EnvVar.OD_SETTINGS_FILE, default=EnvVar.DEFAULT__OD_SETTINGS_FILE)
)


def read_settings(settings_file: str = SETTINGS_FILE) -> AppSettings:
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
        logger.error(
            f"[bold red]Configuration file {settings_file} not found.[/bold red]"
        )
    except Exception as e:
        logger.exception(
            f"[bold red]Error loading settings from {settings_file}: {e}[/bold red]"
        )
    return None


def save_settings(
    settings: AppSettings,
    config_dir: str = CONF_DIR,
    settings_file: str = SETTINGS_FILE,
) -> bool:
    """
    Save the application settings to the configuration file.

    Returns:
        bool: True if settings were saved successfully, False otherwise.
    """
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(settings_file, "w") as f:
            f.write(settings.model_dump_json(indent=4))
        logger.info(f"[bold green]Settings saved to {settings_file}[/bold green]")
        return True
    except Exception as e:
        logger.exception(
            f"[bold red]Error saving settings to {settings_file}: {e}[/bold red]"
        )
        return False


def get_httpx_client(verify: bool, timeout: float) -> httpx.Client:
    """
    Obtain an HTTPX client for making requests.

    Args:
        verify (bool): Whether to verify SSL certificates.
        timeout (float): The timeout for requests in seconds.

    Returns:
        httpx.Client: An HTTPX client configured with the specified settings.
    """
    if verify is False:
        logger.warning(
            "SSL verification is disabled. This is not recommended for production use."
        )
    ctx = ssl.create_default_context(
        cafile=os.environ.get("SSL_CERT_FILE", default=certifi.where()),
        capath=os.environ.get("SSL_CERT_DIR", default=None),
    )
    client = httpx.Client(
        verify=verify if (verify is not None and verify is False) else ctx,
        follow_redirects=True,
        trust_env=True,
        http2=True,
        timeout=timeout,
        headers={"User-Agent": user_agent},
    )
    return client


def cleanup_unnecessary_files(unnecessary_files: Set[str]):
    """
    Cleans up unnecessary files and directories.

    Args:
        unnecessary_files (Set[str]): A set of file paths to be removed.
    """
    list_of_unnecessary_files = list(unnecessary_files)
    unnecessary_directories = set()
    for file_object in list_of_unnecessary_files:
        try:
            if not os.path.isdir(file_object):
                os.remove(file_object)
                logger.info(f"Removed unnecessary file: {file_object}")
            else:
                # If it's a directory, we don't remove it yet because it may not be empty.
                unnecessary_directories.add(file_object)
            unnecessary_files.remove(file_object)
        except Exception as e:
            logger.error(f"Failed to remove unnecessary file {file_object}: {e}")

    # Now remove unnecessary directories if they are empty
    for directory in unnecessary_directories:
        try:
            os.rmdir(directory)
            logger.info(f"Removed unnecessary directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to remove unnecessary directory {directory}: {e}")
