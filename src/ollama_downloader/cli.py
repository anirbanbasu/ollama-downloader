import os
import sys
import tempfile
import certifi
from typing_extensions import Annotated
import typer
import httpx
import ssl
import hashlib

from rich import print as print
from rich import print_json as printj

from ollama_downloader.data_models import AppSettings, ImageManifest
from ollama_downloader.common import ic

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)


def _get_settings() -> AppSettings:
    """Load settings from the configuration file."""
    try:
        with open("conf/settings.json", "r") as f:
            return AppSettings.model_validate_json(f.read())
    except FileNotFoundError:
        print("Configuration file not found. Please create 'conf/settings.json'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading settings: {e}")
        sys.exit(1)


def _get_httpx_client(verify: bool, timeout: float) -> httpx.Client:
    """
    Obtain an HTTPX client for making requests to the Frankfurter API.
    """
    if verify is False:
        print(
            "[yellow]SSL verification is disabled. This is not recommended for production use.[/yellow]"
        )
    ctx = ssl.create_default_context(
        cafile=os.environ.get("SSL_CERT_FILE", certifi.where()),
        capath=os.environ.get("SSL_CERT_DIR"),
    )
    client = httpx.Client(
        verify=verify if (verify is not None and verify is False) else ctx,
        follow_redirects=True,
        trust_env=True,
        timeout=timeout,
    )
    return client


def _download_file_from_url(client: httpx.Client, url: str) -> None:
    """
    Download a file from a URL and save it to the specified destination.
    """
    try:
        response = client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            temp_file.close()
        print(f"Downloaded {url} to {temp_file.name}")
        ic(hashlib.sha256(response.content).hexdigest())
    except httpx.HTTPStatusError as e:
        print(f"Failed to download {url}: {e}")
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        sys.exit(1)
    finally:
        if temp_file.name and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
            print(f"Temporary file {temp_file.name} removed.")


@app.command()
def hello() -> None:
    """Print a hello message."""
    print("Hello from ollama-downloader!")
    settings = _get_settings()
    ic(settings)


@app.command()
def download(
    model: Annotated[
        str, typer.Argument(help="The name of the model to download, e.g., llama3.1")
    ],
    tag: Annotated[
        str, typer.Argument(help="The tag of the model to download, e.g., latest")
    ] = "latest",
) -> None:
    """Download a model from the Ollama server."""
    settings = _get_settings()
    manifest_url = f"{settings.ollama_storage.registry_base_url}{model}/manifests/{tag}"
    print(
        f"Downloading and validating model manifest: [bold cyan]{model}:{tag}[/bold cyan] from {manifest_url}"
    )
    with _get_httpx_client(
        verify=settings.ollama_storage.verify_ssl,
        timeout=settings.ollama_storage.timeout,
    ) as http_client:
        try:
            response = http_client.get(manifest_url)
            response.raise_for_status()
            # This is not the right way to create the digest, but it is a placeholder
            ic(hashlib.sha256(response.content).hexdigest())
            manifest = ImageManifest.model_validate_json(response.text, strict=False)
            printj(manifest.model_dump_json())
            for layer in manifest.layers:
                print(
                    f"Layer: [bold cyan]{layer.mediaType}[/bold cyan], Size: [bold green]{layer.size}[/bold green] bytes, Digest: [bold yellow]{layer.digest}[/bold yellow]"
                )
                print(
                    f"BLOB URL: {settings.ollama_storage.registry_base_url}{model}/blobs/{layer.digest.replace(':', '-')}"
                )
                if "model" not in layer.mediaType:
                    print(f"Downloading layer [bold cyan]{layer.digest}[/bold cyan]")
                    _download_file_from_url(
                        http_client,
                        f"{settings.ollama_storage.registry_base_url}{model}/blobs/{layer.digest.replace(':', '-')}",
                    )
        except httpx.HTTPStatusError as e:
            print(f"Failed to download model manifest: {e}")
            sys.exit(1)
        except httpx.RequestError as e:
            print(f"Request error: {e}")
            sys.exit(1)


def main() -> None:
    try:
        app()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
