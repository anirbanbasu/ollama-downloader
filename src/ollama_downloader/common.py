import logging
import os
import ssl
from typing import Set

import certifi
import httpx
import platform

from rich.logging import RichHandler

from environs import env

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, markup=True)],
)

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str("LOG_LEVEL", default="INFO").upper())

UA_NAME_VER = env.str("OD_UA_NAME_VER", default="ollama-downloader/0.1.0")
user_agent = f"{UA_NAME_VER} ({platform.platform()} {platform.system()}-{platform.release()} Python-{platform.python_version()})"

CONF_DIR = env.str("OD_CONF_DIR", default="conf")
SETTINGS_FILE = os.path.join(
    CONF_DIR, env.str("OD_SETTINGS_FILE", default="settings.json")
)


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
