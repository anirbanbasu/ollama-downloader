from abc import ABC, abstractmethod
import logging
import os
import platform
import ssl
from typing import Set

import certifi
import httpx

from environs import env

from ollama_downloader.common import EnvVar

logger = logging.getLogger(__name__)


class Downloader(ABC):
    _cleanup_running: bool = False
    _unnecessary_files: Set[str] = set()
    _user_agent = f"{env.str(EnvVar.OD_UA_NAME_VER, default=EnvVar.DEFAULT__OD_UA_NAME_VER)} \
        ({platform.platform()} {platform.system()}-{platform.release()} Python-{platform.python_version()})"

    @abstractmethod
    def download_model(self, model_identifier: str) -> None:
        """
        Download a supported model into an available Ollama server.

        Args:
            model_identifier (str): The model tag to download, e.g., "gpt-oss:latest" for library models.
            If the tag is omitted, "latest" is assumed. For Hugging Face models, the model identifier is
            of the format <user>/<repository>:<quantisation>, e.g., unsloth/gemma-3-270m-it-GGUF:Q4_K_M.
        """
        pass

    def get_httpx_client(self, verify: bool, timeout: float) -> httpx.Client:
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
            cafile=env.str("SSL_CERT_FILE", default=certifi.where()),
            capath=env.str("SSL_CERT_DIR", default=None),
        )
        client = httpx.Client(
            verify=verify if (verify is not None and verify is False) else ctx,
            follow_redirects=True,
            trust_env=True,
            http2=True,
            timeout=timeout,
            headers={"User-Agent": self._user_agent},
        )
        return client

    def cleanup_unnecessary_files(self):
        """
        Cleans up unnecessary files and directories created during downloading models.
        """
        # TODO: Is this thread-safe? Should we use a lock?
        if not self._cleanup_running:
            self._cleanup_running = True
            list_of_unnecessary_files = list(self._unnecessary_files)
            unnecessary_directories = set()
            for file_object in list_of_unnecessary_files:
                try:
                    if not os.path.isdir(file_object):
                        os.remove(file_object)
                        logger.info(f"Removed unnecessary file: {file_object}")
                    else:
                        # If it's a directory, we don't remove it yet because it may not be empty.
                        unnecessary_directories.add(file_object)
                    self._unnecessary_files.remove(file_object)
                except Exception as e:
                    logger.error(
                        f"Failed to remove unnecessary file {file_object}: {e}"
                    )

            # Now remove unnecessary directories if they are empty
            for directory in unnecessary_directories:
                try:
                    os.rmdir(directory)
                    logger.info(f"Removed unnecessary directory: {directory}")
                except OSError as e:
                    logger.error(
                        f"Failed to remove unnecessary directory {directory}: {e}"
                    )
            self._cleanup_running = False
