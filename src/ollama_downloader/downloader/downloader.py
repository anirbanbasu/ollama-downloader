from abc import ABC, abstractmethod


class Downloader(ABC):
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
