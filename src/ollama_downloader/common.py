import grp
import logging
from typing import ClassVar
import platform
import psutil

logger = logging.getLogger(__name__)


class EnvVar:
    LOG_LEVEL = "LOG_LEVEL"
    DEFAULT__LOG_LEVEL = "INFO"

    OD_UA_NAME_VER = "OD_UA_NAME_VER"
    DEFAULT__OD_UA_NAME_VER = "ollama-downloader/0.1.0"

    OD_SETTINGS_FILE = "OD_SETTINGS_FILE"
    DEFAULT__OD_SETTINGS_FILE = "conf/settings.json"


class OllamaSystemInfo:
    """Experimental class to obtain system information related to Ollama."""

    PROCESS_NAME: ClassVar[str] = "ollama"
    _instance: ClassVar = None

    _os_name: str = ""
    _process_id: int = -1
    process_env_vars: dict[str, str] = {}
    _parent_process_id: int = -1
    _process_owner: tuple[str, int, str, int] = (
        "",
        -1,
        "",
        -1,
    )  # (username, uid, groupname, gid)
    _listening_on: str = ""  # e.g. "http://localhost:11434"
    _models_dir_path: str = ""
    _likely_daemon: bool = False

    def __new__(cls: type["OllamaSystemInfo"]) -> "OllamaSystemInfo":
        """
        The singleton pattern is used to ensure that calls to certain methods are cached
        using instance variables. However, only one instance of this class should exist.
        """
        if cls._instance is None:
            # Create instance using super().__new__ to bypass any recursion
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance

    def is_windows(self) -> bool:
        """Check if the operating system is Windows."""
        if self._os_name == "":
            self._os_name = platform.system().lower()
        return self._os_name == "windows"

    def is_macos(self) -> bool:
        """Check if the operating system is macOS."""
        if self._os_name == "":
            self._os_name = platform.system().lower()
        return self._os_name == "darwin"

    def is_running(self) -> bool:
        """
        Check if the Ollama process is running on the system.
        This will not work if Ollama is running in a container.
        """
        if self._process_id == -1:
            for proc in psutil.process_iter(["pid", "name"]):
                if (
                    proc.info["name"]
                    and
                    # Should we check as == or in?
                    OllamaSystemInfo.PROCESS_NAME.lower() == proc.info["name"].lower()
                ):
                    self._process_id = proc.info["pid"]
                    break
            if self._process_id != -1:
                logger.debug(f"Ollama process found with PID {self._process_id}.")
                try:
                    self.process_env_vars = {}
                    proc = psutil.Process(self._process_id)
                    # FIXME: These will not capture any variables that the Ollama process sets after it starts.
                    # For example, "OLLAMA_MODELS" is not captured this way unless explicitly passed.
                    self.process_env_vars.update(proc.environ())
                except psutil.NoSuchProcess:
                    ...
                except psutil.AccessDenied:
                    logger.debug(
                        f"Environment variables of {proc.name()} ({self._process_id}) cannot be retrieved. Perhaps, {proc.name()} is running as a different user."
                    )
        return self._process_id != -1

    def get_parent_process_id(self) -> int:
        """
        Get the parent process ID of the Ollama process.
        This will fail if Ollama is running as a service and this function is called by not a super-user.
        """
        if self._parent_process_id == -1 and self.is_running():
            try:
                proc = psutil.Process(self._process_id)
                self._parent_process_id = proc.ppid()
            except psutil.NoSuchProcess:
                ...
            except psutil.AccessDenied:
                logger.debug(
                    f"Parent process ID of {proc.name} ({self._process_id}) cannot be retrieved. Perhaps, {proc.name} is running as a service."
                )
        return self._parent_process_id

    def get_process_owner(self) -> tuple[str, int, str, int] | None:
        """Get the owner of the Ollama process as a tuple of (username, uid, groupname, gid)."""
        if self._process_owner == ("", -1, "", -1) and self.is_running():
            try:
                proc = psutil.Process(self._process_id)
                username = proc.username()
                uid = proc.uids().real if hasattr(proc, "uids") else -1
                gid = proc.gids().real if hasattr(proc, "gids") else -1
                groupname = grp.getgrgid(gid).gr_name if gid != -1 else ""
                self._process_owner = (username, uid, groupname, gid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                ...
        return self._process_owner

    def infer_listening_on(self) -> str | None:
        """Get the address and port the Ollama process is listening on."""
        if self._listening_on == "" and self.is_running():
            try:
                proc = psutil.Process(self._process_id)
                for conn in proc.net_connections(kind="inet"):
                    if conn.status == psutil.CONN_LISTEN:
                        # TODO: Are we considering IPv6 or assuming that it will always be available over IPv4?
                        laddr = (
                            f"{conn.laddr.ip}:{conn.laddr.port}"
                            if conn.laddr.ip != "::"
                            else f"127.0.0.1:{conn.laddr.port}"
                        )
                        self._listening_on = f"http://{laddr}"
                        # Just take the first listening address
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                ...
        return self._listening_on

    def infer_models_dir_path(self) -> str | None:
        """Get the path to the models directory used by Ollama."""
        raise NotImplementedError("This method is not implemented yet.")

    def is_likely_daemon(self) -> bool:
        """Infer if the Ollama process is likely running as a daemon/service."""
        raise NotImplementedError("This method is not implemented yet.")
