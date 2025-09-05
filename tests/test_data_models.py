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
