from ollama_downloader.data.data_models import AppSettings, OllamaLibrary, OllamaServer


class TestDataModels:
    """
    Class to group tests related to data models.
    """

    def test_app_settings(self):
        """
        Test the AppSettings data model.
        """
        settings = AppSettings.load_or_create_default()
        assert settings is not None
        assert isinstance(settings.ollama_server, OllamaServer)
        assert isinstance(settings.ollama_library, OllamaLibrary)

        another_settings = AppSettings()
        yet_another_settings = AppSettings()
        # Singleton behavior
        assert id(settings) == id(another_settings) == id(yet_another_settings)

        another_settings.ollama_library.verify_ssl = False
        # Singleton behavior check: changing one instance changes all
        assert settings.ollama_library.verify_ssl is False

    def test_app_settings_invalid_urls(self):
        """
        Test the AppSettings data model parts with invalid URLs.
        """
        invalid_urls = [
            "htp://invalid-url",  # Invalid scheme
            "http://",  # Missing host
            "http://invalid-url:port",  # Invalid port
            "http://invalid-url/path with spaces",  # Spaces in URL
            "http://inva|id-url",  # Invalid character
        ]

        for url in invalid_urls:
            try:
                OllamaServer(url=url)
                assert False, f"Expected validation error for URL: {url}"
            except Exception as e:
                assert "ValidationError" in str(type(e)) or "AssertionError" in str(
                    type(e)
                )

            try:
                OllamaLibrary(library_base_url=url, registry_base_url=url)
                assert False, f"Expected validation error for URL: {url}"
            except Exception as e:
                assert "ValidationError" in str(type(e)) or "AssertionError" in str(
                    type(e)
                )

    def test_app_settings_invalid_path(self):
        """
        Test the OllamaLibrary data model with an invalid models_path.
        """
        invalid_paths = [
            "/path/that/does/not/exist",
            "/etc/passwd",  # Not a directory
            "https://not.a.path",  # Not a directory path
        ]

        for path in invalid_paths:
            try:
                OllamaLibrary(models_path=path)
                assert False, f"Expected validation error for path: {path}"
            except Exception as e:
                assert "ValidationError" in str(type(e)) or "AssertionError" in str(
                    type(e)
                )
