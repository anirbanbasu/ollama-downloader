import os
import pytest
from ollama_downloader.common import OllamaSystemInfo


class TestOllamaSystemInfo:
    """
    Class to group tests related to the OllamaSystemInfo class.
    Assume that Ollama is NOT running as a service/daemon for these tests.
    """

    def test_is_running(self):
        system_info = OllamaSystemInfo()
        assert system_info.is_running() is True

    def test_access_env_vars(self):
        system_info = OllamaSystemInfo()
        assert isinstance(system_info.process_env_vars, dict)
        # Assuming that the environment variable "PATH" is set in the Ollama process
        assert "PATH" in system_info.process_env_vars
        # assert "OLLAMA_MODELS" in system_info.process_env_vars

    def test_process_owner(self):
        system_info = OllamaSystemInfo()
        owner_info = system_info.get_process_owner()
        assert isinstance(owner_info, tuple)
        assert len(owner_info) == 4  # (username, uid, groupname, gid)
        assert isinstance(owner_info[0], str)  # username
        assert isinstance(owner_info[1], int)  # uid
        assert isinstance(owner_info[2], str)  # groupname
        assert isinstance(owner_info[3], int)  # gid

    def test_get_parent_process_id(self):
        system_info = OllamaSystemInfo()
        parent_pid = system_info.get_parent_process_id()
        assert isinstance(parent_pid, int)
        assert parent_pid > 0  # Parent PID should be a positive integer

    @pytest.mark.skip(reason="This feature is currently under development.")
    def test_is_likely_daemon(self):
        system_info = OllamaSystemInfo()
        assert system_info.is_likely_daemon() is False

    @pytest.mark.skip(reason="This feature is currently under development.")
    def test_infer_models_dir_path(self):
        system_info = OllamaSystemInfo()
        models_path = system_info.infer_models_dir_path()
        assert models_path is not None
        # Check that the inferred models path exists and is a directory
        assert os.path.exists(models_path)
        assert os.path.isdir(models_path)
