import logging
from environs import env
from rich.logging import RichHandler

from ollama_downloader.common import EnvVar

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

logging.basicConfig(
    level=env.str(EnvVar.LOG_LEVEL, default=EnvVar.DEFAULT__LOG_LEVEL).upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=False, markup=True, show_path=False, show_time=False
        )
    ],
)
