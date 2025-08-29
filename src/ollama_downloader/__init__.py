import logging
from environs import env
from rich.logging import RichHandler

from ollama_downloader.common import EnvVar

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
