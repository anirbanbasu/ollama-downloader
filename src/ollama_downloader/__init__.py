import logging
import os
from environs import Env
from marshmallow.validate import OneOf
from rich.logging import RichHandler

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

env = Env()
if os.path.exists(".env"):  # This check is only necessary for Nuitka-compiled binaries.
    env.read_env()  # Read .env file, if it exists


class EnvVar:
    LOG_LEVEL = env.str(
        "LOG_LEVEL",
        default="info",
        validate=OneOf(["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    ).upper()

    OD_UA_NAME_VER = env.str("OD_UA_NAME_VER", default="ollama-downloader/0.1.1")

    OD_SETTINGS_FILE = env.str(
        "OD_SETTINGS_FILE", default=os.path.join("conf", "settings.json")
    )


logging.basicConfig(
    level=EnvVar.LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=False, markup=True, show_path=False, show_time=False
        )
    ],
)
