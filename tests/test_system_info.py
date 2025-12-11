import os

import httpx

from ollama_downloader.sysinfo import OllamaSystemInfo


class TestOllamaSystemInfo:
    """Class to group tests related to the OllamaSystemInfo class.

    Assume that Ollama is NOT running as a service/daemon for these tests.
    """

    def test_is_running(self):
        """Test if the Ollama process is running."""
        system_info = OllamaSystemInfo()
        assert system_info.is_running() is True

    def test_is_not_windows(self):
        """Test that the operating system is not Windows."""
        system_info = OllamaSystemInfo()
        assert system_info.is_windows() is False

    def test_is_linux_or_macos(self):
        """Test that the operating system is either Linux or macOS."""
        system_info = OllamaSystemInfo()
        # Both need to be invoked to ensure coverage
        is_linux = system_info.is_linux()
        is_macos = system_info.is_macos()
        assert is_linux or is_macos  # Only one of them should be True

    def test_access_env_vars(self):
        """Test accessing the environment variables of the Ollama process."""
        system_info = OllamaSystemInfo()
        assert isinstance(system_info.process_env_vars, dict)
        # Assuming that the environment variable "PATH" is set in the Ollama process
        assert "PATH" in system_info.process_env_vars
        # assert "OLLAMA_MODELS" in system_info.process_env_vars

    def test_process_owner(self):
        """Test retrieving the process owner information."""
        system_info = OllamaSystemInfo()
        owner_info = system_info.get_process_owner()
        assert isinstance(owner_info, tuple)
        assert len(owner_info) == 4  # (username, uid, groupname, gid)
        assert isinstance(owner_info[0], str)  # username
        assert owner_info[0] != ""  # username should not be empty
        assert isinstance(owner_info[1], int)  # uid
        assert owner_info[1] > 0  # uid should be a positive integer
        assert isinstance(owner_info[2], str)  # groupname
        assert owner_info[2] != ""  # groupname should not be empty
        assert isinstance(owner_info[3], int)  # gid
        assert owner_info[3] > 0  # gid should be a positive integer
        assert (
            owner_info[0] != "ollama" and owner_info[2] != "ollama"
        )  # Assuming Ollama is not running as user/group "ollama"

    def test_get_parent_process_id(self):
        """Test retrieving the parent process ID of the Ollama process."""
        system_info = OllamaSystemInfo()
        parent_pid = system_info.get_parent_process_id()
        assert isinstance(parent_pid, int)
        assert parent_pid > 0  # Parent PID should be a positive integer

    def test_infer_listening_on(self):
        """Test inferring the listening address of the Ollama process."""
        system_info = OllamaSystemInfo()
        listening_on = system_info.infer_listening_on()
        assert listening_on != ""
        # Try connecting and check the response
        with httpx.Client(timeout=30.0) as client:
            response = client.get(listening_on)
            assert response.status_code == 200
            assert response.text == "Ollama is running"

    # @pytest.mark.skip(reason="Disabled temporarily.")
    def test_is_model_dir_env_var_set(self):
        """Test checking if the 'OLLAMA_MODELS' environment variable is set in the Ollama process."""
        system_info = OllamaSystemInfo()
        # Assuming that the environment variable "OLLAMA_MODELS" has not been passed to the Ollama process
        assert system_info.is_model_dir_env_var_set() is False

    # @pytest.mark.skip(reason="This feature is currently under development.")
    def test_is_likely_daemon(self):
        """Test checking if Ollama is likely running as a daemon/service."""
        system_info = OllamaSystemInfo()
        assert system_info.is_likely_daemon() is False

    # @pytest.mark.skip(reason="This feature is currently under development.")
    def test_infer_models_dir_path(self):
        """Test inferring the models directory path used by Ollama."""
        system_info = OllamaSystemInfo()
        models_path = os.path.expanduser(system_info.infer_models_dir_path())
        assert models_path is not None
        if models_path != "":
            # Check that the inferred models path exists and is a directory
            assert os.path.exists(models_path)
            assert os.path.isdir(models_path)
