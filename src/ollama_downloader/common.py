import grp
import logging
import os
import re
from typing import ClassVar
import platform
import psutil

from ollama import Client as OllamaClient

logger = logging.getLogger(__name__)


class EnvVar:
    LOG_LEVEL = "LOG_LEVEL"
    DEFAULT__LOG_LEVEL = "INFO"

    OD_UA_NAME_VER = "OD_UA_NAME_VER"
    DEFAULT__OD_UA_NAME_VER = "ollama-downloader/0.1.1"

    OD_SETTINGS_FILE = "OD_SETTINGS_FILE"
    DEFAULT__OD_SETTINGS_FILE = "conf/settings.json"


class OllamaSystemInfo:
    """Experimental class to obtain system information related to Ollama."""

    PROCESS_NAME: ClassVar[str] = "ollama"
    _instance: ClassVar = None

    os_name: str = ""
    process_id: int = -1
    process_env_vars: dict[str, str] = {}
    parent_process_id: int = -1
    process_owner: tuple[str, int, str, int] = (
        "",
        -1,
        "",
        -1,
    )  # (username, uid, groupname, gid)
    listening_on: str = ""  # e.g. "http://localhost:11434"
    models_dir_path: str = ""
    likely_daemon: bool = False

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
        if self.os_name == "":
            self.os_name = platform.system()
        return self.os_name.lower() == "windows"

    def is_macos(self) -> bool:
        """Check if the operating system is macOS."""
        if self.os_name == "":
            self.os_name = platform.system()
        return self.os_name.lower() == "darwin"

    def is_running(self) -> bool:
        """
        Check if the Ollama process is running on the system.
        This will not work if Ollama is running in a container.
        """
        if self.process_id == -1:
            for proc in psutil.process_iter(["pid", "name"]):
                if (
                    proc.info["name"]
                    and
                    # Use exact match to avoid false positives from substring matches but what about Windows, e.g., "ollama.exe"?
                    OllamaSystemInfo.PROCESS_NAME.lower() == proc.info["name"].lower()
                ):
                    self.process_id = proc.info["pid"]
                    break
            if self.process_id != -1:
                logger.debug(f"Ollama process found with PID {self.process_id}.")
                try:
                    self.process_env_vars = {}
                    proc = psutil.Process(self.process_id)
                    # FIXME: These will not capture any variables that the Ollama process sets after it starts.
                    # For example, "OLLAMA_MODELS" is not captured this way unless explicitly passed.
                    self.process_env_vars.update(proc.environ())
                    if len(self.process_env_vars) > 0:
                        logger.debug(
                            f"{len(self.process_env_vars)} environment variables of process {proc.name()} ({self.process_id}, {proc.status()}) were obtained."
                        )
                except psutil.NoSuchProcess:
                    ...
                except psutil.AccessDenied:
                    logger.warning(
                        f"Environment variables of process {proc.name()} ({self.process_id}, {proc.status()}) cannot be retrieved. Run auto-config as super-user."
                    )
            else:
                logger.warning(
                    "Ollama process not found. Maybe, it is not installed or it is not running."
                )
        return self.process_id != -1

    def get_parent_process_id(self) -> int:
        """
        Get the parent process ID of the Ollama process.
        This will fail if Ollama is running as a service and this function is called by not a super-user.
        """
        if self.parent_process_id == -1 and self.is_running():
            try:
                proc = psutil.Process(self.process_id)
                self.parent_process_id = proc.ppid()
            except psutil.NoSuchProcess:
                ...
            except psutil.AccessDenied:
                logger.warning(
                    f"Parent process ID of process {proc.name()} ({self.process_id}, {proc.status()}) cannot be retrieved. Run auto-config as super-user."
                )
        return self.parent_process_id

    def get_process_owner(self) -> tuple[str, int, str, int] | None:
        """Get the owner of the Ollama process as a tuple of (username, uid, groupname, gid)."""
        if self.process_owner == ("", -1, "", -1) and self.is_running():
            try:
                proc = psutil.Process(self.process_id)
                username = proc.username()
                uid = proc.uids().real if hasattr(proc, "uids") else -1
                gid = proc.gids().real if hasattr(proc, "gids") else -1
                groupname = grp.getgrgid(gid).gr_name if gid != -1 else ""
                self.process_owner = (username, uid, groupname, gid)
                logger.debug(
                    f"Owner of process {proc.name()} ({self.process_id}, {proc.status()}): {self.process_owner}"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                ...
        return self.process_owner

    def is_model_dir_env_var_set(self) -> bool:
        """Check if the environment variable 'OLLAMA_MODELS' is set in the Ollama process."""
        if self.process_env_vars:
            return "OLLAMA_MODELS" in self.process_env_vars
        return False

    def infer_listening_on(self) -> str | None:
        """Get the address and port the Ollama process is listening on."""
        if self.listening_on == "" and self.is_running():
            try:
                proc = psutil.Process(self.process_id)
                for conn in proc.net_connections(kind="inet"):
                    if conn.status == psutil.CONN_LISTEN:
                        # TODO: Are we considering IPv6 or assuming that it will always be available over IPv4?
                        laddr = (
                            f"{conn.laddr.ip}:{conn.laddr.port}"
                            if conn.laddr.ip != "::"
                            else f"127.0.0.1:{conn.laddr.port}"
                        )
                        self.listening_on = f"http://{laddr}"
                        # Just take the first listening address
                        break
            except psutil.NoSuchProcess:
                ...
            except psutil.AccessDenied:
                logger.warning(
                    f"Parent process ID of process {proc.name()} ({self.process_id}, {proc.status()}) cannot be retrieved. Run auto-config as super-user."
                )
        return self.listening_on

    def infer_models_dir_path(self) -> str:
        """Get the path to the models directory used by Ollama."""
        self.models_dir_path = self.process_env_vars.get("OLLAMA_MODELS", "")
        if self.models_dir_path == "":
            # raise NotImplementedError("This method has been partly implemented only.")
            if self.infer_listening_on() != "":
                ollama_client = OllamaClient(host=self.listening_on)
                models_list = ollama_client.list().models
                if len(models_list) > 0:
                    modelfile_text = ollama_client.show(models_list[0].model).modelfile
                    pattern = re.compile(
                        r"^\s*FROM\s+(.+?)(?:\s*#.*)?$", re.IGNORECASE | re.MULTILINE
                    )
                    match = pattern.search(modelfile_text)
                    if match:
                        model_blob_path = match.group(1).strip()
                        likely_models_dir = str(
                            os.path.dirname(os.path.dirname(model_blob_path))
                        )
                        self.models_dir_path = likely_models_dir.replace(
                            os.path.expanduser("~"), "~"
                        )
                else:
                    logger.warning(
                        "No models are currently installed in Ollama. Cannot infer the models directory path."
                    )
        return self.models_dir_path

    def is_likely_daemon(self) -> bool:
        """Infer if the Ollama process is likely running as a daemon/service."""
        self.get_parent_process_id()
        if self.parent_process_id not in [-1, 1]:
            self.likely_daemon = False
        else:
            proc = psutil.Process(self.process_id)
            if (
                proc.username().lower() in ["ollama", "root"]
                and proc.terminal() is None
            ):
                self.likely_daemon = True
            else:
                self.likely_daemon = False
        return self.likely_daemon
