import datetime
import logging
from typing import List, Set, Tuple

from httpx import URL

from environs import env

from ollama_downloader.common import EnvVar
from ollama_downloader.data.data_models import ImageManifest
from ollama_downloader.downloader.model_downloader import ModelDownloader, ModelSource
import lxml.html
from ollama import Client as OllamaClient


from rich import print as print

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str(EnvVar.LOG_LEVEL, default=EnvVar.DEFAULT__LOG_LEVEL).upper())


class OllamaModelDownloader(ModelDownloader):
    def __init__(self):
        super().__init__()

    def download_model(self, model_identifier: str) -> bool:
        model, tag = (
            model_identifier.split(":")
            if ":" in model_identifier
            else (model_identifier, "latest")
        )
        print(f"Downloading Ollama library model {model}:{tag}")
        # Validate the response as an ImageManifest but don't enforce strict validation
        manifest_json = self._fetch_manifest(
            model_identifier=model_identifier, model_source=ModelSource.OLLAMA
        )
        logger.info(f"Validating manifest for {model}:{tag}")
        manifest = ImageManifest.model_validate_json(manifest_json, strict=True)
        # Keep a list of files to be copied but only copy after all downloads have completed successfully.
        # This is to ensure that we don't copy files that may not be needed if the download fails.
        # Each tuple in the list contains (source_path, named_digest, computed_digest).
        files_to_be_copied: List[Tuple[str, str, str]] = []
        # Download the model configuration BLOB
        logger.info(f"Downloading model configuration {manifest.config.digest}")
        file_model_config, digest_model_config = self._download_model_blob(
            model_identifier=model_identifier,
            named_digest=manifest.config.digest,
            model_source=ModelSource.OLLAMA,
        )
        files_to_be_copied.append(
            (file_model_config, manifest.config.digest, digest_model_config)
        )
        # The layers could be null for cloud-hosted models.
        if manifest.layers:
            for layer in manifest.layers:
                logger.debug(
                    f"Layer: {layer.mediaType}, Size: {layer.size} bytes, Digest: {layer.digest}"
                )
                logger.info(f"Downloading {layer.mediaType} layer {layer.digest}")
                file_layer, digest_layer = self._download_model_blob(
                    model_identifier=model_identifier,
                    named_digest=layer.digest,
                    model_source=ModelSource.OLLAMA,
                )
                files_to_be_copied.append((file_layer, layer.digest, digest_layer))
        # All BLOBs have been downloaded, now copy them to their appropriate destinations.
        for source, named_digest, computed_digest in files_to_be_copied:
            copy_status, copy_destination = self._save_blob(
                source=source,
                named_digest=named_digest,
                computed_digest=computed_digest,
            )
            if copy_status is False:
                raise RuntimeError(
                    f"Failed to copy {named_digest} to {copy_destination}."
                )
        # Finally, save the manifest to its appropriate destination
        self._save_manifest(
            data=manifest_json,
            model_identifier=model_identifier,
            model_source=ModelSource.OLLAMA,
        )
        ts_approximate_manifest_save = datetime.datetime.now()
        # Finally check if it exists in the Ollama
        # Clear the list of unnecessary files before this if errors henceforth are to be tolerated.
        if not self.settings.ollama_server.remove_downloaded_on_error:
            self._unnecessary_files.clear()
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
            print(
                f"Model {found_model.model} successfully downloaded and saved on {found_model.modified_at:%B %d %Y at %H:%M:%S}."
            )
        else:
            raise RuntimeError(
                f"Model {model}:{tag} could not be found in Ollama server after download."
            )
        # If we reached here cleanly, remove all unnecessary file names but don't remove actual files.
        self._unnecessary_files.clear()
        # Finally return success
        return True

    def list_available_models(
        self, page: int | None = None, page_size: int | None = None
    ) -> List[str]:
        if page is not None:
            logger.warning(
                "Pagination is not supported for Ollama library. Ignoring page parameter."
            )
        if page_size is not None:
            logger.warning(
                "Pagination is not supported for Ollama library. Ignoring page_size parameter."
            )
        with self.get_httpx_client(
            verify=self.settings.ollama_library.verify_ssl,
            timeout=self.settings.ollama_library.timeout,
        ) as client:
            logger.debug(
                f"Updating models list from Ollama library {self.settings.ollama_library.library_base_url}"
            )
            models_response = client.get(self.settings.ollama_library.library_base_url)
            models_response.raise_for_status()
            parsed_models_html = lxml.html.document_fromstring(models_response.text)
            available_models = []
            library_prefix = "/library/"
            for _, attribute, link, _ in lxml.html.iterlinks(parsed_models_html):
                if attribute == "href" and link.startswith(library_prefix):
                    available_models.append(link.replace(library_prefix, ""))
            logger.debug(f"Found {len(available_models)} models in the Ollama library.")
            return available_models

    def list_model_tags(self, model_identifier: str) -> List[str]:
        available_models = self.list_available_models()
        if model_identifier not in available_models:
            raise RuntimeError(
                f"Model {model_identifier} not found in the library models list."
            )
        with self.get_httpx_client(
            verify=self.settings.ollama_library.verify_ssl,
            timeout=self.settings.ollama_library.timeout,
        ) as client:
            models_tags = []
            logger.debug(
                f"Fetching tags for model {model_identifier} from the Ollama library."
            )
            tags_response = client.get(
                URL(self.settings.ollama_library.library_base_url).join(
                    f"{model_identifier}/tags"
                )
            )
            tags_response.raise_for_status()
            logger.debug(f"Parsing tags for model {model_identifier}.")
            parsed_tags_html = lxml.html.document_fromstring(tags_response.text)
            library_prefix = "/library/"
            named_model_unique_tags: Set[str] = set()
            for _, attribute, link, _ in lxml.html.iterlinks(parsed_tags_html):
                if attribute == "href" and link.startswith(
                    f"{library_prefix}{model_identifier}:"
                ):
                    named_model_unique_tags.add(link.replace(library_prefix, ""))
            models_tags = list(named_model_unique_tags)

            return models_tags
