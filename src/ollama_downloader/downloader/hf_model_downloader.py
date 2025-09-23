import datetime
import logging
from typing import Annotated, List, Optional, Tuple
from urllib.parse import urlparse


from environs import env
from pydantic import Field

from ollama_downloader.common import EnvVar
from ollama_downloader.data.data_models import ImageManifest
from ollama_downloader.downloader.model_downloader import ModelDownloader, ModelSource

# import lxml.html
from ollama import Client as OllamaClient
# from huggingface_hub import configure_http_backend
# import requests  # type: ignore

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str(EnvVar.LOG_LEVEL, default=EnvVar.DEFAULT__LOG_LEVEL).upper())


class HuggingFaceModelDownloader(ModelDownloader):
    def __init__(self):
        super().__init__()
        # if not self.settings.ollama_library.verify_ssl:
        #     logger.warning(
        #         "Disabling SSL verification for HTTP requests. This is not recommended for production use."
        #     )
        #     session = requests.Session()
        #     session.verify = False
        #     configure_http_backend(backend_factory=lambda: session)

    def download_model(self, model_identifier: str) -> bool:
        # Validate the response as an ImageManifest but don't enforce strict validation
        (user, model_repo), quant = (
            model_identifier.split(":")[0].split("/"),
            model_identifier.split(":")[1],
        )
        print(
            f"Downloading Hugging Face model {model_repo} from {user} with {quant} quantisation"
        )
        manifest_json = self._fetch_manifest(
            model_identifier=model_identifier, model_source=ModelSource.HUGGINGFACE
        )
        logger.info(f"Validating manifest for {model_identifier}")
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
            model_source=ModelSource.HUGGINGFACE,
        )
        files_to_be_copied.append(
            (file_model_config, manifest.config.digest, digest_model_config)
        )
        # The layers could be null for cloud-hosted Ollama models but this is here only for consistency.
        if manifest.layers:
            for layer in manifest.layers:
                logger.debug(
                    f"Layer: {layer.mediaType}, Size: {layer.size} bytes, Digest: {layer.digest}"
                )
                logger.info(f"Downloading {layer.mediaType} layer {layer.digest}")
                file_layer, digest_layer = self._download_model_blob(
                    model_identifier=model_identifier,
                    named_digest=layer.digest,
                    model_source=ModelSource.HUGGINGFACE,
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
            model_source=ModelSource.HUGGINGFACE,
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
        search_model = (
            f"{urlparse(ModelDownloader.HF_BASE_URL).hostname}/{model_identifier}"
        )
        for model_info in models_list.models:
            if (
                model_info.model == search_model
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
                f"Model {search_model} could not be found in Ollama server after download."
            )
        # If we reached here cleanly, remove all unnecessary file names but don't remove actual files.
        self._unnecessary_files.clear()
        return True

    def list_available_models(
        self,
        page: Annotated[Optional[int], Field(gt=0)] = None,
        page_size: Annotated[Optional[int], Field(gt=0, le=100)] = None,
    ) -> List[str]:
        page_size = page_size or 100
        next_page = 1
        page = page or 1
        if page_size * (page + 1) >= 1000:
            logger.warning(
                "Hugging Face currently does not allow paging beyond the first 999 models. Follow issue 2741: https://github.com/huggingface/huggingface_hub/issues/2741"
            )
            raise ValueError(
                f"Hugging Face currently does not allow obtaining information beyond the first 999 models. Your requested page {page} with page size {page_size} exceeds this limit by {int(((page + 1) * page_size - 999))} model(s)."
            )
        api_url = f"https://huggingface.co/api/models?apps=ollama&gated=false&limit={page_size}&sort=trendingScore"
        next_page_url = api_url

        with self.get_httpx_client(
            self.settings.ollama_library.verify_ssl,
            self.settings.ollama_library.timeout,
        ) as client:
            while next_page < page and next_page_url:
                models_head = client.head(next_page_url)
                models_head.raise_for_status()
                next_page_url = models_head.links.get("next", {}).get("url")
                next_page += 1
            if next_page_url:
                if next_page > 1:
                    logger.info(f"Requesting page {next_page} from {next_page_url}")
                models_response = client.get(next_page_url)
                models_response.raise_for_status()
            model_identifiers = [
                model["modelId"] for model in list(models_response.json())
            ]

        return model_identifiers

    def list_model_tags(self, model_identifier: str) -> List[str]:
        api_url = f"https://huggingface.co/api/models/{model_identifier}?blobs=true"
        tags = []
        with self.get_httpx_client(
            self.settings.ollama_library.verify_ssl,
            self.settings.ollama_library.timeout,
        ) as client:
            model_response = client.get(api_url)
            model_response.raise_for_status()
            model_info = model_response.json()
            model_siblings = model_info.get("siblings", [])
            for repo_sibling in model_siblings:
                rfilename = repo_sibling.get("rfilename", "")
                if rfilename.endswith(".gguf"):
                    # Try to extract the quantisation from the filename
                    tag = rfilename.split(".gguf")[0].split("-")[-1]
                    tags.append(f"{model_identifier}:{tag}")
        if len(tags) == 0:
            # If no .gguf files found, the model is not for Ollama
            raise RuntimeError(
                f"The model {model_identifier} has no support for Ollama."
            )
        return tags
