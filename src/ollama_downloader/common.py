from enum import StrEnum, auto

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class EnvVar(StrEnum):
    LOG_LEVEL = auto()
    DEFAULT__LOG_LEVEL = "INFO"

    OD_UA_NAME_VER = auto()
    DEFAULT__OD_UA_NAME_VER = "ollama-downloader/0.1.0"

    OD_SETTINGS_FILE = auto()
    DEFAULT__OD_SETTINGS_FILE = "conf/settings.json"
