import os
import sys
import tempfile
import certifi
from typing_extensions import Annotated
import typer
import httpx
import ssl
import hashlib
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn
from rich import print as print
from rich import print_json as printj

from ollama_downloader.data_models import AppSettings, ImageManifest
from ollama_downloader.common import ic

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)

SETTINGS_FILE = "conf/settings.json"


def _get_settings() -> AppSettings:
    """Load settings from the configuration file."""
    try:
        with open(SETTINGS_FILE, "r") as f:
            return AppSettings.model_validate_json(f.read())
    except FileNotFoundError:
        print(f"Configuration file not found. Please create '{SETTINGS_FILE}'.")
        sys.exit(1)


settings = _get_settings()


def _get_manifest_url(model: str, tag: str) -> str:
    """
    Construct the URL for a model manifest based on its name and tag.
    """
    return f"{settings.ollama_storage.registry_base_url}{model}/manifests/{tag}"


def _get_blob_url(model: str, digest: str) -> str:
    """
    Construct the URL for a BLOB based on its digest.
    """
    return f"{settings.ollama_storage.registry_base_url}{model}/blobs/{digest.replace(':', '-')}"


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
        http2=True,
        timeout=timeout,
    )
    return client


def _download_model_blob(model: str, digest: str) -> None:
    """
    Download a file given the digest and save it to the specified destination.
    """
    url = _get_blob_url(model=model, digest=digest)
    try:
        sha256_hash = hashlib.new("sha256")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            with _get_httpx_client(
                settings.ollama_storage.verify_ssl, settings.ollama_storage.timeout
            ).stream("GET", url) as response:
                response.raise_for_status()
                ic(response.headers)
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
        print(f"Downloaded {url} to {temp_file.name}")
        content_digest = sha256_hash.hexdigest()
        ic(content_digest)
        if digest[7:] != content_digest:
            print(
                f"[red]Digest mismatch: expected {digest[7:]}, got {content_digest}[/red]"
            )
        else:
            print(
                f"[green]Digest verified: {content_digest} matches expected digest.[/green]"
            )
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
    manifest_url = _get_manifest_url(model=model, tag=tag)
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
            manifest = ImageManifest.model_validate_json(response.text, strict=False)
            printj(manifest.model_dump_json())
            print(
                f"Downloading manifest [bold cyan]{manifest.config.digest}[/bold cyan]"
            )
            _download_model_blob(
                model=model,
                digest=manifest.config.digest,
            )
            for layer in manifest.layers:
                print(
                    f"Layer: [bold cyan]{layer.mediaType}[/bold cyan], Size: [bold green]{layer.size}[/bold green] bytes, Digest: [bold yellow]{layer.digest}[/bold yellow]"
                )
                print(f"Downloading layer [bold cyan]{layer.digest}[/bold cyan]")
                _download_model_blob(
                    model=model,
                    digest=layer.digest,
                )
        except httpx.HTTPStatusError as e:
            print(f"Failed to download model manifest: {e}")
            sys.exit(1)
        except httpx.RequestError as e:
            print(f"Request error: {e}")
            sys.exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
