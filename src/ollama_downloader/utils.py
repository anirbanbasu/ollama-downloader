import logging
import os
import ssl
from typing import Set

import certifi
import httpx
import platform

from environs import env
from ollama_downloader.common import EnvVar

logger = logging.getLogger(__name__)

user_agent = f"{env.str(EnvVar.OD_UA_NAME_VER, default=EnvVar.DEFAULT__OD_UA_NAME_VER)} ({platform.platform()} {platform.system()}-{platform.release()} Python-{platform.python_version()})"


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
