import logging
import os
import ssl

import certifi
import httpx

from rich.logging import RichHandler


try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, markup=True)],
)

# Initialize the logger
logger = logging.getLogger(__name__)


def get_httpx_client(verify: bool, timeout: float) -> httpx.Client:
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
