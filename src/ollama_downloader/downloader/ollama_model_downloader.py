import datetime
import json
import logging
import os
from typing import Dict, List, Set, Tuple

from httpx import URL

from environs import env

from ollama_downloader.common import EnvVar
from ollama_downloader.data.data_models import ImageManifest
from ollama_downloader.downloader.model_downloader import ModelDownloader, ModelSource
import lxml.html
from ollama import Client as OllamaClient


from rich import print as print
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
)

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str(EnvVar.LOG_LEVEL, default=EnvVar.DEFAULT__LOG_LEVEL).upper())


class OllamaModelDownloader(ModelDownloader):
    def __init__(self):
        super().__init__()
        self.models_tags: Dict[str, list] = {}
        self.library_models: List[str] = []

    def _save_models_tags_cache(
        self,
    ) -> None:
        """
        Save the models tags cache to a file.
        """
        if hasattr(self, "models_tags"):
            with open(self.settings.ollama_library.models_tags_cache, "w") as f:
                f.write(json.dumps(self.models_tags, indent=2))
        else:
            logger.warning("No models tags cache to save.")

    def load_models_tags_cache(self) -> None:
        """
        Load the models tags cache from a file.
        """
        if os.path.exists(self.settings.ollama_library.models_tags_cache):
            with open(self.settings.ollama_library.models_tags_cache, "r") as f:
                self.models_tags = json.loads(f.read())
            logger.info(
                f"Loaded models tags cache from {self.settings.ollama_library.models_tags_cache}"
            )
            self.library_models = list(self.models_tags.keys())
        else:
            logger.warning(
                f"No models tags cache found at {self.settings.ollama_library.models_tags_cache}"
            )

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

    def update_models_list(self) -> list:
        """
        Update the list of models available in the Ollama library.

        Returns:
            list: A list of model names available in the Ollama library.
        """
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
            self.library_models = []
            library_prefix = "/library/"
            for _, attribute, link, _ in lxml.html.iterlinks(parsed_models_html):
                if attribute == "href" and link.startswith(library_prefix):
                    self.library_models.append(link.replace(library_prefix, ""))
            logger.debug(
                f"Found {len(self.library_models)} models in the Ollama library."
            )
            return self.library_models

    def list_models_tags(
        self, model: str | None = None, update: bool = False
    ) -> dict[str, list]:
        """
        Update the tags for each model or a named model in the Ollama library.

        Returns:
            dict[str, list]: A dictionary where keys are model names and values are lists of tags.
        """
        self.load_models_tags_cache()
        if not hasattr(self, "library_models") or update:
            self.update_models_list()
        _base_url = URL(self.settings.ollama_library.library_base_url)
        with self.get_httpx_client(
            verify=self.settings.ollama_library.verify_ssl,
            timeout=self.settings.ollama_library.timeout,
        ) as client:
            if model is not None:
                if model not in self.library_models:
                    logger.error(f"Model {model} not found in the library models list.")
                    return {}
                else:
                    if not hasattr(self, "models_tags") or update:
                        logger.debug(
                            f"Fetching tags for model {model} from the Ollama library."
                        )
                        tags_response = client.get(_base_url.join(f"{model}/tags"))
                        tags_response.raise_for_status()
                        logger.debug(f"Parsing tags for model {model}.")
                        parsed_tags_html = lxml.html.document_fromstring(
                            tags_response.text
                        )
                        library_prefix = "/library/"
                        named_model_unique_tags: Set[str] = set()
                        for _, attribute, link, _ in lxml.html.iterlinks(
                            parsed_tags_html
                        ):
                            if attribute == "href" and link.startswith(
                                f"{library_prefix}{model}:"
                            ):
                                if model not in self.models_tags:
                                    self.models_tags[model] = []
                                named_model_unique_tags.add(
                                    link.replace(library_prefix, "")
                                )
                        self.models_tags[model] = list(named_model_unique_tags)
                        logger.debug(f"Updating tags for model {model} in the cache.")
                        self._save_models_tags_cache()
                    return {model: self.models_tags.get(model, [])}
            else:
                if (
                    not hasattr(self, "models_tags")
                    or update
                    or len(self.models_tags) == 0
                ):
                    # A full update has been requested
                    model_counter = 0
                    with Progress(
                        TextColumn(text_format="{task.description}"),
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        BarColumn(bar_width=None),
                        MofNCompleteColumn(),
                    ) as progress:
                        tags_task = progress.add_task(
                            "Updating models",
                            total=len(self.library_models),
                        )
                        for m in self.library_models:
                            tags_response = client.get(_base_url.join(f"{m}/tags"))
                            tags_response.raise_for_status()
                            parsed_tags_html = lxml.html.document_fromstring(
                                tags_response.text
                            )
                            model_unique_tags: Set[str] = set()
                            for _, attribute, link, _ in lxml.html.iterlinks(
                                parsed_tags_html
                            ):
                                if attribute == "href" and link.startswith(
                                    f"/library/{m}:"
                                ):
                                    if m not in self.models_tags:
                                        self.models_tags[m] = []
                                    model_unique_tags.add(link.replace("/library/", ""))
                            self.models_tags[m] = list(model_unique_tags)
                            model_counter += 1
                            progress.update(tags_task, completed=model_counter)
                    logger.info(
                        f"Updated {len(self.models_tags)} models with tags from the Ollama library."
                    )
                    self._save_models_tags_cache()
                return self.models_tags
