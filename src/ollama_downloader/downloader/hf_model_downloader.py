import datetime
import inspect
import logging
from typing import List, Tuple
from urllib.parse import urlparse

from importlib.metadata import version

from environs import env

from ollama_downloader.common import EnvVar
from ollama_downloader.data.data_models import ImageManifest
from ollama_downloader.downloader.model_downloader import ModelDownloader, ModelSource

# import lxml.html
from ollama import Client as OllamaClient
from huggingface_hub import HfApi, configure_http_backend
import requests  # type: ignore

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str(EnvVar.LOG_LEVEL, default=EnvVar.DEFAULT__LOG_LEVEL).upper())


class HuggingFaceModelDownloader(ModelDownloader):
    def __init__(self):
        super().__init__()
        if not self.settings.ollama_library.verify_ssl:
            logger.warning(
                "Disabling SSL verification for HTTP requests. This is not recommended for production use."
            )
            session = requests.Session()
            session.verify = False
            configure_http_backend(backend_factory=lambda: session)

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
        return found_model

    def list_available_models(self) -> List[str]:
        hf_api = HfApi()
        # Temporary check to see if `apps` parameter is available in the current version of huggingface-hub
        # Enforce huggingface-hub>=0.34.5 in pyproject.toml in the future.
        method_signature = inspect.signature(hf_api.list_models)
        hf_api_version = version("huggingface-hub")
        limit_results_to = 100
        if "apps" in method_signature.parameters:
            # Limit to 100 models for brevity and find a better way of doing this later.
            # TODO: See: https://github.com/huggingface/huggingface_hub/issues/2741
            models = hf_api.list_models(
                apps="ollama", gated=False, limit=limit_results_to
            )
            model_identifiers = [model.modelId for model in list(models)]

            if "dev" in hf_api_version and hf_api_version.startswith("0.35"):
                logger.warning(
                    f"You are using a development version of huggingface-hub: {hf_api_version}. Please upgrade to the latest release to use the apps parameter in the future."
                )
            logger.warning(
                f"Listing models from Hugging Face is currently limited to the top {limit_results_to} models only. Browse through the full list at https://huggingface.co/models?apps=ollama&gated=False"
            )
            return model_identifiers
        raise NotImplementedError(
            "Listing models from Hugging Face while filtering by supported applications, e.g., Ollama is not implemented yet. Follow issue 3319: https://github.com/huggingface/huggingface_hub/issues/3319"
        )

    def list_model_tags(self, model_identifier: str) -> List[str]:
        hf_api = HfApi()
        model_info = hf_api.model_info(repo_id=model_identifier, files_metadata=True)
        tags = []
        for repo_sibling in model_info.siblings:
            if repo_sibling.rfilename.endswith(".gguf"):
                # Try to extract the quantisation from the filename
                tag = repo_sibling.rfilename.split(".gguf")[0].split("-")[-1]
                tags.append(f"{model_identifier}:{tag}")
        if len(tags) == 0:
            # If no .gguf files found, the model is not for Ollama
            raise RuntimeError(
                f"The model {model_identifier} has no support for Ollama."
            )
        return tags
