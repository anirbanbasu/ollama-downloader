import datetime
import hashlib
import os
import shutil
import tempfile
from typing import List, Set, Tuple
from urllib.parse import urlparse

from ollama_downloader.data_models import AppSettings, ImageManifest
from ollama_downloader.common import (
    logger,
    # read_settings,
    # write_settings,
    get_httpx_client,
)
from ollama import Client as OllamaClient

from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn


class OllamaModelDownloader:
    def __init__(self, settings: AppSettings | None = None):
        """
        Initialize the model downloader with application settings.

        Args:
            settings (AppSettings | None): The application settings. If None, defaults will be used.
        """
        self.settings = settings or AppSettings()
        if not self.settings.read_settings():
            self.settings.save_settings()
        self.unnecessary_files: Set[str] = set()

    def _cleanup_unnecessary_files(self):
        """
        Clean up temporary files created during the download process.
        These could include files downloaded but need to be removed because the entire model download failed or was interrupted.
        """
        list_of_unnecessary_files = list(self.unnecessary_files)
        unnecessary_directories = set()
        for file_object in list_of_unnecessary_files:
            try:
                if not os.path.isdir(file_object):
                    os.remove(file_object)
                    logger.info(f"Removed unnecessary file: {file_object}")
                else:
                    # If it's a directory, we don't remove it yet because it may not be empty.
                    unnecessary_directories.add(file_object)
                self.unnecessary_files.remove(file_object)
            except Exception as e:
                logger.error(f"Failed to remove unnecessary file {file_object}: {e}")

        # Now remove unnecessary directories if they are empty
        for directory in unnecessary_directories:
            try:
                os.rmdir(directory)
                logger.info(f"Removed unnecessary directory: {directory}")
            except OSError as e:
                logger.error(f"Failed to remove unnecessary directory {directory}: {e}")

    def _get_manifest_url(self, model: str, tag: str) -> str:
        """
        Construct the URL for a model manifest based on its name and tag.

        Args:
            model (str): The name of the model, e.g., llama3.1.
            tag (str): The tag of the model, e.g., latest.

        Returns:
            str: The URL for the model manifest.
        """
        logger.debug(f"Constructing manifest URL for {model}:{tag}")
        return (
            f"{self.settings.ollama_storage.registry_base_url}{model}/manifests/{tag}"
        )

    def _get_blob_url(self, model: str, digest: str) -> str:
        """
        Construct the URL for a BLOB based on its digest.

        Args:
            model (str): The name of the model, e.g., llama3.1.
            digest (str): The digest of the BLOB prefixed with the digest algorithm followed by a colon character.

        Returns:
            str: The URL for the BLOB.
        """
        logger.debug(f"Constructing BLOB URL for {model} with digest {digest}")
        return f"{self.settings.ollama_storage.registry_base_url}{model}/blobs/{digest.replace(':', '-')}"

    def _fetch_manifest(self, model: str, tag: str) -> str:
        """
        Fetch the manifest for a model from the Ollama registry.

        Args:
            model (str): The name of the model, e.g., llama3.1.
            tag (str): The tag of the model, e.g., latest.

        Returns:
            str: The JSON string of the model manifest.
        """
        url = self._get_manifest_url(model=model, tag=tag)
        logger.info(f"Downloading manifest for [bold cyan]{model}:{tag}[/bold cyan]")
        with get_httpx_client(
            self.settings.ollama_storage.verify_ssl,
            self.settings.ollama_storage.timeout,
        ) as http_client:
            response = http_client.get(url)
            response.raise_for_status()
            return response.text

    def _download_model_blob(self, model: str, named_digest: str) -> tuple:
        """
        Download a file given the digest and save it to the specified destination.

        Args:
            model (str): The name of the model, e.g., llama3.1.
            named_digest (str): The digest of the BLOB prefixed with the digest algorithm followed by a colon character.

        Returns:
            tuple: A tuple containing the path to the downloaded file and its computed SHA256 digest.
        """
        url = self._get_blob_url(model=model, digest=named_digest)
        # try:
        sha256_hash = hashlib.new("sha256")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.unnecessary_files.add(temp_file.name)
            with get_httpx_client(
                self.settings.ollama_storage.verify_ssl,
                self.settings.ollama_storage.timeout,
            ).stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers["Content-Length"])

                with Progress(
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    BarColumn(bar_width=None),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                ) as progress:
                    download_task = progress.add_task("Download", total=total)
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

    def _save_manifest_to_destination(
        self,
        data: str,
        model: str,
        tag: str,
    ) -> str:
        """
        Copy a manifest data to destination.

        Args:
            data (str): The JSON string of the manifest.
            model (str): The name of the model, e.g., llama3.1.
            tag (str): The tag of the model, e.g., latest.

        Returns:
            str: The path to the saved manifest file.
        """
        ollama_registry_host = urlparse(
            self.settings.ollama_storage.registry_base_url
        ).hostname
        manifests_toplevel_dir = os.path.join(
            (
                os.path.expanduser(self.settings.ollama_storage.models_path)
                if self.settings.ollama_storage.models_path.startswith("~")
                else self.settings.ollama_storage.models_path
            ),
            "manifests",
        )
        manifests_dir = os.path.join(
            manifests_toplevel_dir,
            ollama_registry_host,
            "library",
            model,
        )
        if not os.path.exists(manifests_dir):
            logger.warning(
                f"Manifests path {manifests_dir} does not exist. Will attempt to create it."
            )
            os.makedirs(manifests_dir)
            self.unnecessary_files.add(manifests_dir)
        target_file = os.path.join(manifests_dir, tag)
        with open(target_file, "w") as f:
            f.write(data)
            logger.info(f"Saved manifest to {target_file}")
        if self.settings.ollama_storage.user_group:
            user, group = self.settings.ollama_storage.user_group
            shutil.chown(target_file, user, group)
            # The directory ownership must also be changed because it may have been created by a different user, most likely a sudoer
            # TODO: Is this necessary or can the ownership change to the top-level directory cascade down?
            shutil.chown(manifests_dir, user, group)
            shutil.chown(manifests_toplevel_dir, user, group)
            logger.info(
                f"Changed ownership of {target_file} to user: {user}, group: {group}"
            )
        self.unnecessary_files.add(target_file)
        return target_file

    def _copy_blob_to_destination(
        self,
        source: str,
        named_digest: str,
        computed_digest: str,
    ) -> tuple:
        """
        Copy a downloaded BLOB to the destination and verify its digest.

        Args:
            source (str): The path to the downloaded BLOB.
            named_digest (str): The expected digest of the BLOB prefixed with the digest algorithm followed by the colon character.
            computed_digest (str): The computed digest of the BLOB.

        Returns:
            tuple: A tuple containing a boolean indicating success and the path to the copied file.
        """
        if computed_digest != named_digest[7:]:
            logger.error(
                f"Digest mismatch: expected {named_digest[7:]}, got {computed_digest}"
            )
            return False, None
        blobs_dir = os.path.join(
            (
                os.path.expanduser(self.settings.ollama_storage.models_path)
                if self.settings.ollama_storage.models_path.startswith("~")
                else self.settings.ollama_storage.models_path
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
        logger.info(f"Moved {source} to {target_file}")
        if self.settings.ollama_storage.user_group:
            user, group = self.settings.ollama_storage.user_group
            shutil.chown(target_file, user, group)
            shutil.chown(blobs_dir, user, group)
            # Set permissions to rw-r-----
            os.chmod(target_file, 0o640)
            logger.info(
                f"Changed ownership of {target_file} to user: {user}, group: {group}"
            )
        self.unnecessary_files.remove(source)
        self.unnecessary_files.add(target_file)
        return True, target_file

    def download_model(self, model: str, tag: str = "latest") -> None:
        # Implementation of the model downloading logic
        """Download a model from the Ollama server."""
        # Validate the response as an ImageManifest but don't enforce strict validation
        manifest_json = self._fetch_manifest(model=model, tag=tag)
        logger.debug(f"Validating manifest for {model}:{tag}")
        manifest = ImageManifest.model_validate_json(manifest_json, strict=False)
        logger.info(
            f"Downloading model configuration [bold cyan]{manifest.config.digest}[/bold cyan]"
        )
        # Keep a list of files to be copied but only copy after all downloads have completed successfully.
        # This is to ensure that we don't copy files that may not be needed if the download fails.
        # Each tuple in the list contains (source_path, named_digest, computed_digest).
        files_to_be_copied: List[Tuple[str, str, str]] = []
        # Download the model configuration BLOB
        file_model_config, digest_model_config = self._download_model_blob(
            model=model,
            named_digest=manifest.config.digest,
        )
        files_to_be_copied.append(
            (file_model_config, manifest.config.digest, digest_model_config)
        )
        for layer in manifest.layers:
            logger.debug(
                f"Layer: [bold cyan]{layer.mediaType}[/bold cyan], Size: [bold green]{layer.size}[/bold green] bytes, Digest: [bold yellow]{layer.digest}[/bold yellow]"
            )
            logger.info(
                f"Downloading [bold cyan]{layer.mediaType}[/bold cyan] layer [bold cyan]{layer.digest}[/bold cyan]"
            )
            file_layer, digest_layer = self._download_model_blob(
                model=model,
                named_digest=layer.digest,
            )
            files_to_be_copied.append((file_layer, layer.digest, digest_layer))
        # All BLOBs have been downloaded, now copy them to their appropriate destinations.
        for source, named_digest, computed_digest in files_to_be_copied:
            copy_status, copy_destination = self._copy_blob_to_destination(
                source=source,
                named_digest=named_digest,
                computed_digest=computed_digest,
            )
            if copy_status is False:
                logger.error(f"Failed to copy {named_digest} to {copy_destination}.")
        # Finally, save the manifest to its appropriate destination
        self._save_manifest_to_destination(
            data=manifest_json,
            model=model,
            tag=tag,
        )
        ts_approximate_manifest_save = datetime.datetime.now()
        # Finally check if it exists in the Ollama
        ollama_client = OllamaClient(
            host=self.settings.ollama_server.url,
            # timeout=self.settings.ollama_server.timeout,
            # TODO: Add API key authentication logic
        )
        models_list = ollama_client.list()
        found_model = None
        for model_info in models_list.models:
            if (
                model_info.model == f"{model}:{tag}"
                # TODO: Is this timestamp assumption right that the listing is completed within a minute of saving?
                and abs(
                    model_info.modified_at.replace(tzinfo=None)
                    - ts_approximate_manifest_save
                )
                < datetime.timedelta(minutes=1)
            ):
                found_model = model_info
                break
        if found_model:
            logger.info(
                f"[green]Model [bold]{found_model.model}[/bold] successfully downloaded and saved on {found_model.modified_at:%B %d %Y at %H:%M:%S}.[/green]"
            )
        else:
            raise RuntimeError(
                f"Model {model}:{tag} could not be found in Ollama server after download."
            )
        # If we reached here cleanly, remove all unnecessary file names but don't remove actual files.
        self.unnecessary_files.clear()
