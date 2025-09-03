from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import logging
import os
import platform
import shutil
import ssl
import tempfile
from typing import List, Set, Tuple
from urllib.parse import urlparse

from ollama import Client as OllamaClient

import certifi
import httpx

from environs import env

from ollama_downloader.common import EnvVar

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
)

from ollama_downloader.data.data_models import AppSettings

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    OLLAMA = 1
    HUGGINGFACE = 2


class ModelDownloader(ABC):
    HF_BASE_URL = "https://hf.co/v2/"

    def __init__(self):
        self.settings: AppSettings = AppSettings.load_or_create_default()
        if not self.settings:
            raise RuntimeError("Failed to load or create and save default settings.")
        self._user_agent: str = f"{env.str(EnvVar.OD_UA_NAME_VER, default=EnvVar.DEFAULT__OD_UA_NAME_VER)} ({platform.platform()} {platform.system()}-{platform.release()} Python-{platform.python_version()})"
        self._unnecessary_files: Set[str] = set()
        self._cleanup_running: bool = False

    @abstractmethod
    def download_model(self, model_identifier: str) -> bool:
        """
        Download a supported model into an available Ollama server.

        Args:
            model_identifier (str): The model tag to download, e.g., "gpt-oss:latest" for library models.
            If the tag is omitted, "latest" is assumed. For Hugging Face models, the model identifier is
            of the format <user>/<repository>:<quantisation>, e.g., unsloth/gemma-3-270m-it-GGUF:Q4_K_M.

        Returns:
            bool: True if the model was successfully downloaded and verified, False otherwise.
        """
        pass

    @abstractmethod
    def list_available_models(
        self, page: int | None = None, page_size: int | None = None
    ) -> List[str]:
        """
        List available models. If pagination is supported by the source, page and page_size can be used to control the results.
        If pagination is not supported or the page and page_size are None, the number of models will be returned will depend on
        the implementing sub-class.

        Args:
            page (int | None): The page number to retrieve, if pagination is supported.
            page_size (int | None): The number of items per page, if pagination is supported

        Returns:
            list[str]: A list of available model identifiers, excluding any of their tags.
        """
        pass

    @abstractmethod
    def list_model_tags(self, model_identifier: str) -> List[str]:
        """
        List available tags for a specific model.

        Args:
            model_identifier (str): The name of the model to list tags for, e.g., "gpt-oss" for an Ollama library model
            or "unsloth/gemma-3-270m-it-GGUF" for a Hugging Face model.

        Returns:
            list[str]: A list of available tags for the specified model, e.g., ["latest", "20b"] or
            the Hugging Face quantisations such as ["Q4_K_M", "Q4_K_S"].
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

    def _make_manifest_url(
        self, model_identifier: str, model_source: ModelSource
    ) -> httpx.URL:
        """
        Constructs the manifest URL for a given model identifier.

        Args:
            model_identifier (str): The model identifier, e.g., "gpt-oss:latest" for an Ollama library model
            or "unsloth/gemma-3-270m-it-GGUF:Q4_K_M" for a Hugging Face model.
            model_source (ModelSource): The source of the model (e.g., OLLAMA or HUGGINGFACE).

        Returns:
            httpx.URL: The constructed manifest URL.
        """
        match model_source:
            case ModelSource.OLLAMA:
                model, tag = (
                    model_identifier.split(":")
                    if ":" in model_identifier
                    else (model_identifier, "latest")
                )
                logger.debug(f"Constructing manifest URL for {model}:{tag}")
                return httpx.URL(self.settings.ollama_library.registry_base_url).join(
                    f"{model}/manifests/{tag}"
                )
            case ModelSource.HUGGINGFACE:
                logger.debug(f"Constructing manifest URL for {model_identifier}")
                return httpx.URL(
                    f"{ModelDownloader.HF_BASE_URL}{model_identifier.replace(':', '/manifests/')}"
                )
            case _:
                raise ValueError(f"Unsupported model source: {model_source}")

    def _fetch_manifest(self, model_identifier: str, model_source: ModelSource) -> str:
        """
        Fetches the manifest JSON content for a given model identifier.

        Args:
            model_identifier (str): The model identifier, e.g., "gpt-oss:latest" for an Ollama library model
            or "unsloth/gemma-3-270m-it-GGUF:Q4_K_M" for a Hugging Face model.
            model_source (ModelSource): The source of the model (e.g., OLLAMA or HUGGINGFACE).

        Returns:
            str: The fetched manifest JSON content as a string.
        """
        url = self._make_manifest_url(model_identifier, model_source)
        logger.info(f"Downloading manifest from {url}")
        with self.get_httpx_client(
            self.settings.ollama_library.verify_ssl,
            self.settings.ollama_library.timeout,
        ) as http_client:
            response = http_client.get(url)
            response.raise_for_status()
            return response.text

    def _make_blob_url(
        self, model_identifier: str, digest: str, model_source: ModelSource
    ) -> httpx.URL:
        """
        Constructs the blob URL for a given model and digest.

        Args:
            model_identifier (str): The model name, e.g., "gpt-oss:latest".
            digest (str): The digest of the blob, e.g., "sha256:abcdef...".
            model_source (ModelSource): The source of the model (e.g., OLLAMA or HUGGINGFACE).

        Returns:
            httpx.URL: The constructed blob URL.
        """
        logger.debug(
            f"Constructing blob URL for {model_identifier} with digest {digest}"
        )
        model_name = model_identifier.split(":")[0]
        match model_source:
            case ModelSource.OLLAMA:
                return httpx.URL(self.settings.ollama_library.registry_base_url).join(
                    f"{model_name}/blobs/{digest.replace(':', '-')}"
                )
            case ModelSource.HUGGINGFACE:
                return httpx.URL(
                    f"{ModelDownloader.HF_BASE_URL}{model_name}/blobs/{digest}"
                )
            case _:
                raise ValueError(f"Unsupported model source: {model_source}")

    def _download_model_blob(
        self, model_identifier: str, named_digest: str, model_source: ModelSource
    ) -> tuple:
        """
        Downloads a model blob given its named digest.

        Args:
            model_identifier (str): The model name, e.g., "gpt-oss:latest".
            named_digest (str): The named digest of the blob, e.g., "sha256:abcdef...".
            model_source (ModelSource): The source of the model (e.g., OLLAMA or HUGGINGFACE).

        Returns:
            tuple: A tuple containing the path to the downloaded file as string and its digest.
        """
        url = self._make_blob_url(
            model_identifier=model_identifier,
            digest=named_digest,
            model_source=model_source,
        )
        # try:
        sha256_hash = hashlib.new("sha256")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self._unnecessary_files.add(temp_file.name)
            with self.get_httpx_client(
                self.settings.ollama_library.verify_ssl,
                self.settings.ollama_library.timeout,
            ).stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers["Content-Length"])

                with Progress(
                    TextColumn(text_format="{task.description}"),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    BarColumn(bar_width=None),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                ) as progress:
                    download_task = progress.add_task(
                        f"Downloading BLOB {named_digest[:11]}...{named_digest[-4:]}",
                        total=total,
                    )
                    for chunk in response.iter_bytes():
                        sha256_hash.update(chunk)
                        temp_file.write(chunk)
                        progress.update(
                            download_task, completed=response.num_bytes_downloaded
                        )
        logger.debug(f"Downloaded {url} to {temp_file.name}")
        content_digest = sha256_hash.hexdigest()
        logger.debug(f"Computed SHA256 digest of {temp_file.name}: {content_digest}")
        return temp_file.name, content_digest

    def _save_manifest(
        self, data: str, model_identifier: str, model_source: ModelSource
    ):
        """
        Saves the manifest data to the appropriate location based on the model source.

        Args:
            data (str): The manifest JSON content as a string.
            model_identifier (str): The model identifier, e.g., "gpt-oss:latest" for an Ollama library model
            or "unsloth/gemma-3-270m-it-GGUF:Q4_K_M" for a Hugging Face model.
            model_source (ModelSource): The source of the model (e.g., OLLAMA or HUGGINGFACE).

        Returns:
            str: The path to the saved manifest file.
        """
        manifests_toplevel_dir = os.path.join(
            (
                os.path.expanduser(self.settings.ollama_library.models_path)
                if self.settings.ollama_library.models_path.startswith("~")
                else self.settings.ollama_library.models_path
            ),
            "manifests",
        )
        model_identifier_splits = model_identifier.split(":")
        manifests_dir = None
        model_source_host = None
        match model_source:
            case ModelSource.OLLAMA:
                model_source_host = urlparse(
                    self.settings.ollama_library.registry_base_url
                ).hostname
                manifests_dir = os.path.join(
                    manifests_toplevel_dir,
                    model_source_host,
                    "library",
                    model_identifier_splits[0],
                )
            case ModelSource.HUGGINGFACE:
                model_source_host = urlparse(ModelDownloader.HF_BASE_URL).hostname
                manifests_dir = os.path.join(
                    manifests_toplevel_dir,
                    # FIXME: This is a hack for mypy -- the model_source_host should not be None
                    model_source_host or "",
                    model_identifier_splits[0],
                )
            case _:
                raise ValueError(f"Unsupported model source: {model_source}")
        if not os.path.exists(manifests_dir):
            logger.warning(
                f"Manifests path {manifests_dir} does not exist. Will attempt to create it."
            )
            os.makedirs(manifests_dir)
            self._unnecessary_files.add(manifests_dir)
        target_file = os.path.join(manifests_dir, model_identifier_splits[1])
        with open(target_file, "w") as f:
            f.write(data)
            logger.info(f"Saved manifest to {target_file}")
        if self.settings.ollama_library.user_group:
            user, group = self.settings.ollama_library.user_group
            shutil.chown(target_file, user, group)
            # The directory ownership must also be changed because it may have been created by a different user, most likely a sudoer
            # TODO: Is this necessary or can the ownership change to the top-level directory cascade down?
            shutil.chown(manifests_dir, user, group)
            shutil.chown(manifests_toplevel_dir, user, group)
            logger.info(
                f"Changed ownership of {target_file} to user: {user}, group: {group}"
            )
        self._unnecessary_files.add(target_file)
        return target_file

    def _save_blob(
        self,
        source: str,
        named_digest: str,
        computed_digest: str,
    ) -> Tuple[bool, str | None]:
        """
        Saves the downloaded blob to the appropriate location based on the model source.

        Args:
            source (str): The path to the downloaded BLOB.
            named_digest (str): The expected digest of the BLOB prefixed with the digest algorithm followed by the colon character.
            computed_digest (str): The computed digest of the BLOB.

        Returns:
            Tuple[bool, str | None]: A tuple containing a boolean indicating success or failure,
            and the path to the saved BLOB if successful, None otherwise.
        """
        if computed_digest != named_digest[7:]:
            logger.error(
                f"Digest mismatch: expected {named_digest[7:]}, got {computed_digest}"
            )
            return False, None

        blobs_dir = os.path.join(
            (
                os.path.expanduser(self.settings.ollama_library.models_path)
                if self.settings.ollama_library.models_path.startswith("~")
                else self.settings.ollama_library.models_path
            ),
            "blobs",
        )

        logger.info(f"BLOB {named_digest} digest verified successfully.")
        if not os.path.isdir(blobs_dir):
            logger.error(f"BLOBS path {blobs_dir} must be a directory.")
            return False, None
        if not os.path.exists(blobs_dir):
            logger.error(f"BLOBS path {blobs_dir} must exist.")
            return False, None
        target_file = os.path.join(blobs_dir, named_digest.replace(":", "-"))
        shutil.move(source, target_file)
        self._unnecessary_files.remove(source)
        self._unnecessary_files.add(target_file)
        logger.info(f"Moved {source} to {target_file}")
        if self.settings.ollama_library.user_group:
            user, group = self.settings.ollama_library.user_group
            shutil.chown(target_file, user, group)
            shutil.chown(blobs_dir, user, group)
            # Set permissions to rw-r-----
            os.chmod(target_file, 0o640)
            logger.info(
                f"Changed ownership of {target_file} to user: {user}, group: {group}"
            )
        return True, target_file

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

    def remove_model(self, model_identifier: str) -> bool:
        """
        Removes a model from the Ollama server.
        """
        ollama_client = OllamaClient(
            host=self.settings.ollama_server.url,
            # timeout=self.settings.ollama_server.timeout,
            # TODO: Add API key authentication logic
        )
        search_model = (
            f"{urlparse(ModelDownloader.HF_BASE_URL).hostname}/{model_identifier}"
        )
        response = ollama_client.delete(search_model)
        if hasattr(response, "status") and response.status == "success":
            logger.info(
                f"Successfully removed model {model_identifier} from Ollama server at {self.settings.ollama_server.url}."
            )
            return True
        else:
            logger.error(
                f"Failed to remove model {model_identifier} from Ollama server at {self.settings.ollama_server.url}."
            )
            return False
