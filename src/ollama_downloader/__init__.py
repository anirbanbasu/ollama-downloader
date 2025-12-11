import logging
import os

from environs import Env
from marshmallow.validate import OneOf
from rich.logging import RichHandler

env = Env()
# This check is only necessary for Nuitka-compiled binaries.
if os.path.exists(".env"):
    # Read .env file, if it exists
    env.read_env()  # pragma: no cover


class EnvVar:
    """Environment Variables for Ollama Downloader."""

    OD_LOG_LEVEL = env.str(
        "OD_LOG_LEVEL",
        default="info",
        validate=OneOf(["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    ).upper()

    OD_UA_NAME_VER = env.str("OD_UA_NAME_VER", default="ollama-downloader/0.1.1")

    OD_SETTINGS_FILE = env.str("OD_SETTINGS_FILE", default=os.path.join("conf", "settings.json"))


logging.basicConfig(
    level=EnvVar.OD_LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, markup=True, show_path=False, show_time=False)],
)
