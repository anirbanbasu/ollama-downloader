import datetime
import os
import shutil
import sys
import logging
import tempfile
import certifi
from typing_extensions import Annotated
import typer
import httpx
import ssl
from urllib.parse import urlparse
import hashlib
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn
from rich.logging import RichHandler
from rich import print as print
from rich import print_json as printj

from ollama import Client as OllamaClient

from ollama_downloader.data_models import AppSettings, ImageManifest

# Initialize the Typer application
app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A command-line interface for the Ollama downloader.",
)

# Path to the settings file
SETTINGS_FILE = "conf/settings.json"


def _get_settings() -> AppSettings:
    """
    Load settings from the configuration file.

    Returns:
        AppSettings: The application settings loaded from the configuration file.
    """
    try:
        with open(SETTINGS_FILE, "r") as f:
            # Parse the JSON file into the AppSettings model
            parsed_settings = AppSettings.model_validate_json(f.read())
        return parsed_settings
    except FileNotFoundError:
        logger.exception(
            f"[bold red]Configuration file not found. Please create '{SETTINGS_FILE}'.[/bold red]"
        )
        return None
    except Exception as e:
        logger.exception(f"[bold red]Error loading settings: {e}[/bold red]")
        return None


# Global settings variable
settings: AppSettings = None

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, markup=True)],
)

# Initialize the logger
logger = logging.getLogger(__name__)


def _get_manifest_url(model: str, tag: str) -> str:
    """
    Construct the URL for a model manifest based on its name and tag.

    Args:
        model (str): The name of the model, e.g., llama3.1.
        tag (str): The tag of the model, e.g., latest.

    Returns:
        str: The URL for the model manifest.
    """
    logger.debug(f"Constructing manifest URL for {model}:{tag}")
    return f"{settings.ollama_storage.registry_base_url}{model}/manifests/{tag}"


def _get_blob_url(model: str, digest: str) -> str:
    """
    Construct the URL for a BLOB based on its digest.

    Args:
        model (str): The name of the model, e.g., llama3.1.
        digest (str): The digest of the BLOB prefixed with the digest algorithm followed by a colon character.

    Returns:
        str: The URL for the BLOB.
    """
    logger.debug(f"Constructing BLOB URL for {model} with digest {digest}")
    return f"{settings.ollama_storage.registry_base_url}{model}/blobs/{digest.replace(':', '-')}"


def _get_httpx_client(verify: bool, timeout: float) -> httpx.Client:
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
        cafile=os.environ.get("SSL_CERT_FILE", default=certifi.where()),
        capath=os.environ.get("SSL_CERT_DIR", default=None),
    )
    client = httpx.Client(
        verify=verify if (verify is not None and verify is False) else ctx,
        follow_redirects=True,
        trust_env=True,
        http2=True,
        timeout=timeout,
    )
    return client


def _download_model_blob(model: str, digest: str) -> tuple:
    """
    Download a file given the digest and save it to the specified destination.

    Args:
        model (str): The name of the model, e.g., llama3.1.
        digest (str): The digest of the BLOB prefixed with the digest algorithm followed by a colon character.

    Returns:
        tuple: A tuple containing the path to the downloaded file and its computed SHA256 digest.
    """
    url = _get_blob_url(model=model, digest=digest)
    # try:
    sha256_hash = hashlib.new("sha256")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with _get_httpx_client(
            settings.ollama_storage.verify_ssl, settings.ollama_storage.timeout
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
    ollama_registry_host = urlparse(settings.ollama_storage.registry_base_url).hostname
    manifests_toplevel_dir = os.path.join(
        (
            os.path.expanduser(settings.ollama_storage.models_path)
            if settings.ollama_storage.models_path.startswith("~")
            else settings.ollama_storage.models_path
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
    target_file = os.path.join(manifests_dir, tag)
    with open(target_file, "w") as f:
        f.write(data)
        logger.info(f"Saved manifest to {target_file}")
    if settings.ollama_storage.user_group:
        user, group = settings.ollama_storage.user_group
        shutil.chown(target_file, user, group)
        # The directory ownership must also be changed because it may have been created by a different user, most likely a sudoer
        shutil.chown(manifests_dir, user, group)
        shutil.chown(manifests_toplevel_dir, user, group)
        logger.info(
            f"Changed ownership of {target_file} to user: {user}, group: {group}"
        )
    return target_file


def _copy_blob_to_destination(
    source: str,
    digest: str,
    computed_digest: str,
    move_instead_of_copy: bool = True,
) -> tuple:
    """
    Copy a downloaded BLOB to the destination and verify its digest.

    Args:
        source (str): The path to the downloaded BLOB.
        destination (str): The path where the BLOB should be copied.
        digest (str): The expected digest of the BLOB.
        computed_digest (str): The computed digest of the BLOB.
        move_instead_of_copy (bool): Whether to move the file instead of copying it.

    Returns:
        tuple: A tuple containing a boolean indicating success and the path to the copied file.
    """
    if computed_digest != digest[7:]:
        logger.error(f"Digest mismatch: expected {digest[7:]}, got {computed_digest}")
        return False, None
    blobs_dir = os.path.join(
        (
            os.path.expanduser(settings.ollama_storage.models_path)
            if settings.ollama_storage.models_path.startswith("~")
            else settings.ollama_storage.models_path
        ),
        "blobs",
    )
    logger.info(f"BLOB {digest} digest verified successfully.")
    if not os.path.isdir(blobs_dir):
        logger.error(f"BLOBS path {blobs_dir} must be a directory.")
        return False, None
    if not os.path.exists(blobs_dir):
        logger.error(f"BLOBS path {blobs_dir} must exist.")
        return False, None
    target_file = os.path.join(blobs_dir, digest.replace(":", "-"))
    if move_instead_of_copy:
        shutil.move(source, target_file)
        logger.info(f"Moved {source} to {target_file}")
    else:
        shutil.copyfile(source, target_file)
        logger.info(f"Copied {source} to {target_file}")
    if settings.ollama_storage.user_group:
        user, group = settings.ollama_storage.user_group
        shutil.chown(target_file, user, group)
        shutil.chown(blobs_dir, user, group)
        # Set permissions to rw-r-----
        os.chmod(target_file, 0o640)
        logger.info(
            f"Changed ownership of {target_file} to user: {user}, group: {group}"
        )
    return True, target_file


@app.command()
def show_config():
    """Shows the application configuration as JSON."""
    printj(settings.model_dump_json())


@app.command()
def download(
    model: Annotated[
        str, typer.Argument(help="The name of the model to download, e.g., llama3.1")
    ],
    tag: Annotated[
        str, typer.Argument(help="The tag of the model to download, e.g., latest")
    ] = "latest",
):
    """Download a model from the Ollama server."""
    manifest_url = _get_manifest_url(model=model, tag=tag)
    logger.info(
        f"Downloading and validating manifest for [bold cyan]{model}:{tag}[/bold cyan]"
    )
    with _get_httpx_client(
        verify=settings.ollama_storage.verify_ssl,
        timeout=settings.ollama_storage.timeout,
    ) as http_client:
        try:
            response = http_client.get(manifest_url)
            response.raise_for_status()
            # Validate the response as an ImageManifest but don't enforce strict validation
            manifest = ImageManifest.model_validate_json(response.text, strict=False)
            logger.info(
                f"Downloading model configuration [bold cyan]{manifest.config.digest}[/bold cyan]"
            )
            # Download the model configuration BLOB
            file_model_config, digest_model_config = _download_model_blob(
                model=model,
                digest=manifest.config.digest,
            )
            copy_status, copy_destination = _copy_blob_to_destination(
                source=file_model_config,
                digest=manifest.config.digest,
                computed_digest=digest_model_config,
            )
            if copy_status is False:
                logger.error(
                    f"Failed to copy model configuration BLOB {manifest.config.digest} to {copy_destination}."
                )
                sys.exit(1)
            for layer in manifest.layers:
                logger.debug(
                    f"Layer: [bold cyan]{layer.mediaType}[/bold cyan], Size: [bold green]{layer.size}[/bold green] bytes, Digest: [bold yellow]{layer.digest}[/bold yellow]"
                )
                logger.info(
                    f"Downloading [bold cyan]{layer.mediaType}[/bold cyan] layer [bold cyan]{layer.digest}[/bold cyan]"
                )
                file_layer, digest_layer = _download_model_blob(
                    model=model,
                    digest=layer.digest,
                )
                layer_copy_status, layer_copy_destination = _copy_blob_to_destination(
                    source=file_layer,
                    digest=layer.digest,
                    computed_digest=digest_layer,
                )
                if layer_copy_status is False:
                    logger.error(
                        f"Failed to copy model layer {layer.digest} to {layer_copy_destination}."
                    )
                    sys.exit(1)
            # Save the manifest to the destination
            _save_manifest_to_destination(
                data=response.text,
                model=model,
                tag=tag,
            )
            ts_approximate_manifest_save = datetime.datetime.now()
            # Finally check if it exists in the Ollama
            ollama_client = OllamaClient(
                host=settings.ollama_server.url,
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
                logger.error(
                    f"[red]Model [bold]{model}:{tag}[/bold] could not be found in Ollama server after download.[/red]"
                )
                sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"Failed to download model manifest: {e}")
            sys.exit(1)
        except httpx.RequestError as e:
            print(f"Request error: {e}")
            sys.exit(1)


def main():
    """Main entry point for the CLI application."""
    # Load the application settings and initialize the global settings variable
    global settings
    settings = _get_settings()
    # If settings are None, we MUST exit the application
    if settings is None:
        sys.exit(1)
    # All good so far, let's start the Typer app
    try:
        app()
    except Exception as e:
        logger.exception(f"[bold red]An error occurred: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
