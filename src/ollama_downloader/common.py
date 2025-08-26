import logging
import platform

from rich.logging import RichHandler

from environs import env

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, markup=True)],
)

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(env.str("LOG_LEVEL", default="INFO").upper())

UA_NAME_VER = env.str("OD_UA_NAME_VER", default="ollama-downloader/0.1.0")
user_agent = f"{UA_NAME_VER} ({platform.platform()} {platform.system()}-{platform.release()} Python-{platform.python_version()})"
