from enum import Enum, auto

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class AllCapsStrEnum(str, Enum):
    # See https://github.com/python/cpython/issues/115509#issuecomment-1946971056
    @staticmethod
    def _generate_next_value_(name, *args):
        return name.upper()


class EnvVar(AllCapsStrEnum):
    LOG_LEVEL = auto()
    DEFAULT__LOG_LEVEL = "INFO"

    OD_UA_NAME_VER = auto()
    DEFAULT__OD_UA_NAME_VER = "ollama-downloader/0.1.0"

    OD_SETTINGS_FILE = auto()
    DEFAULT__OD_SETTINGS_FILE = "conf/settings.json"
