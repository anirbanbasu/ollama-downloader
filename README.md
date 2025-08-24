[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/) [![pytest](https://github.com/anirbanbasu/ollama-downloader/actions/workflows/uv-pytest.yml/badge.svg)](https://github.com/anirbanbasu/ollama-downloader/actions/workflows/uv-pytest.yml)

# Ollama (library and Hugging Face) model downloader

Rather evident from the name, this is a tool to help download models for [Ollama](https://ollama.com/) including [supported models from Hugging Face](https://huggingface.co/models?apps=ollama). However, doesn't Ollama already download models from its library using `ollama pull <model:tag>`?

Yes, but wait, not so fast...!

### How did we get here?

While `ollama pull <model:tag>` certainly works, not always will you get lucky. This is a documented problem, see [issue 941](https://github.com/ollama/ollama/issues/941). The crux of the problem is that Ollama fails to pull a model from its library spitting out an error message as follows.

> Error: digest mismatch, file must be downloaded again: want sha256:1a640cd4d69a5260bcc807a531f82ddb3890ebf49bc2a323e60a9290547135c1, got sha256:5eef5d8ec5ce977b74f91524c0002f9a7adeb61606cdbdad6460e25d58d0f454

People have been facing this for a variety of unrelated reasons and have found specific solutions that perhaps work for only when those specific reasons exist.

[Comment 2989194688](https://github.com/ollama/ollama/issues/941#issuecomment-2989194688) in the issue thread proposes a manual way to download the models from the library. This solution is likely to work more than others.

_Hence, this tool – an automation of that manual process_!

Do note that, as of August 24, 2025, this Ollama downloader can also _download supported models from Hugging Face_!

### Yet another downloader?
Yes, and there exist others, possibly with different purposes.
 - [Ollama-model-downloader](https://github.com/raffaeleguidi/Ollama-model-downloader)
 - [ollama-dl](https://github.com/akx/ollama-dl)
 - [ollama-direct-downloader](https://github.com/Gholamrezadar/ollama-direct-downloader)
 - [ollama-gguf-downloader](https://github.com/olamide226/ollama-gguf-downloader)

## Installation

The directory where you clone this repository will be referred to as the _working directory_ or _WD_ hereinafter.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). To install the project with its minimal dependencies in a virtual environment, run the following in the _WD_. To install all non-essential dependencies (_which are required for developing and testing_), replace the `--no-dev` with the `--all-groups` flag in the following command.

```bash
uv sync --no-dev
```
## Configuration

There will exist, upon execution of the tool, a configuration file `conf/settings.json` in _WD_. It will be created upon the first run. However, you will need to modify it depending on your Ollama installation.

Let's explore the configuration in details. The default content is as follows.

```json
{
    "ollama_server": {
        "url": "http://localhost:11434",
        "api_key": null,
        "remove_downloaded_on_error": true
    },
    "ollama_library": {
        "models_path": "~/.ollama/models",
        "models_tags_cache": "models_tags.json",
        "registry_base_url": "https://registry.ollama.ai/v2/library/",
        "library_base_url": "https://ollama.com/library",
        "verify_ssl": true,
        "timeout": 120.0,
        "user_group": null
    }
}
```

There are two main configuration groups: `ollama_server` and `ollama_library`. The former refers to the server for which you wish to download the model. The latter refers to the Ollama library where the model and related information ought to be downloaded from.

### `ollama_server`

 - The `url` points to the HTTP endpoint of your Ollama server. While the default is http://localhost:11434, note that your Ollama server may actually be running on a different machine, in which case, the URL will have to point to that endpoint correctly.
 - The `api_key` is only necessary if your Ollama server endpoint expects an API key to connect, which is typically not the case.
 - The `remove_downloaded_on_error` is a boolean flag, typically set to `true`. This helps specify whether this downloader tool should remove downloaded files (including temporary files) if it fails to connect to the Ollama server or fails to find the downloaded model.

### `ollama_library`

 - The `models_path` points to the models directory of your Ollama installation. On Linux/UNIX systems, if it has been installed for your own user only then the path is the default `~/.ollama/models`. If it has been installed as a service, however, it could be, for example on Ubuntu 22.04, `/usr/share/ollama/.ollama/models`. Also note that the path could be a network share, if Ollama is on a different machine.
 - The `models_tags_cache` points to the file that will contain the cache of models and their tags as available in the Ollama library, _not your own Ollama installation_.
 - The `registry_base_url` is the URL to the Ollama registry. Unless you have a custom Ollama registry, use the default value as shown above.
 - Likewise, the `library_base_url` is the URL to the Ollama library. Keep the default value unless you really need to point it to some mirror.
 - The `verify_ssl` is a flag that tells the downloader tool to verify the authenticity of the HTTPS connections it makes to the Ollama registry or the library. Turn this off only if you have a man-in-the-middle proxy with self-signed certificates. Even in that case, typically environment variables `SSL_CERT_FILE` and `SSL_CERT_DIR` can be correctly configured to validate such certificates.
 - The self-explanatory `timeout` specifies the number of seconds to wait before any HTTPS connection to the Ollama registry or library should be allowed to fail.
 - The `user_group` is a specification of the _user_ and the _group_ (as a tuple, e.g., `"user_group": ["user", "group"]`) that owns the path specified by `models_path`. If, for instance, your local Ollama is a service and its model path is `/usr/share/ollama/.ollama/models` then, in order to write to that path, you must run this downloader as _root_. However, the ownership of file objects in that path must be assigned to the user _ollama_ and group _ollama_. If your model path is on a writable network share then you most likely need not specify the user and group.

## Usage
The preferred way to run this downloader is using the `od` script, such as `uv run od --help`.

However, if you need to run it with superuser rights (i.e., using `sudo`) for model download then you should install the script in the `uv` created virtual environment by running `uv pip install -e .` and then you can invoke it as `sudo .venv/bin/od --help`.

The `od` script provides the following commands. All its commands can be listed by running `uv run od --help`.

```bash
 Usage: od [OPTIONS] COMMAND [ARGS]...

 A command-line interface for the Ollama downloader.


╭─ Options ────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────╮
│ show-config      Shows the application configuration as      │
│                  JSON.                                       │
│ list-models      Lists all available models.                 │
│ list-tags        Lists all tags for a specific model.        │
│ model-download   Downloads a specific Ollama model with the  │
│                  given tag.                                  │
╰──────────────────────────────────────────────────────────────╯
```

You can also use `--help` on each command to see command-specific help.

### `show-config`

The `show-config` command simply displays the current configuration from `conf/settings.json`, if it exists. If it does not exist, it creates that file with the default settings and shows the content of that file.

Running `uv run od show-config --help` displays the following.

```bash
Usage: od show-config [OPTIONS]

 Shows the application configuration as JSON.


╭─ Options ────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────╯
```

### `list-models`

The `list-models` command displays an up-to-date list of models that exist in the Ollama library.

Running `uv run od list-models --help` displays the following.

```bash
Usage: od list-models [OPTIONS]

 Lists all available models in the Ollama library.


╭─ Options ────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────╯
```
### `list-tags`

The `list-tags` command shows the tags available for a specified model, or for all models if `model` is not specified. _Note that this command will display cached information unless the `--update` flag is specified._

If you specify the `--update` flag, the cache is updated with newly fetched information from the Ollama library.

Running `uv run od list-tags --help` displays the following.

```bash
Usage: od list-tags [OPTIONS] [MODEL]

 Lists all tags for a specific model.


╭─ Arguments ──────────────────────────────────────────────────╮
│   model      [MODEL]  The name of the model to list tags     │
│                       for, e.g., llama3.1. If not provided,  │
│                       tags of all models will be listed.     │
│                       [default: None]                        │
╰──────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────╮
│ --update    --no-update      Force update the model list and │
│                              its tags before listing.        │
│                              [default: no-update]            │
│ --help                       Show this message and exit.     │
╰──────────────────────────────────────────────────────────────╯
```
### `model-download`

The `model-download` downloads the specified model and its tag from the Ollama library.

During the process of downloading, the following are performed.

1. Validation of the manifest for the specified model and tag.
2. Validation of the SHA256 hash of each downloaded BLOB.
3. Post-download verification with the Ollama server specified by `ollama_server.url` in the configuration that the downloaded model is available.

As an example, run `uv run od model-download all-minilm` to download the `all-minilm:latest` embedding model. _Note that if not specified, the tag is assumed to be `latest`_. You want to specify a tag as `<model>:<tag>`. For instance, run `uv run od model-download llama3.2:3b` to download the `llama3.2` model with the `3b` tag.

Running `uv run od model-download --help` displays the following.

```bash
Usage: od model-download [OPTIONS] MODEL_TAG

 Downloads a specific Ollama model with the given tag.


╭─ Arguments ──────────────────────────────────────────────────╮
│ *    model_tag      TEXT  The name of the model and a        │
│                           specific to download, specified as │
│                           <model>:<tag>, e.g.,               │
│                           llama3.1:8b. If no tag is          │
│                           specified, 'latest' will be        │
│                           assumed.                           │
│                           [default: None]                    │
│                           [required]                         │
╰──────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────╯
```

The following screencast shows the process of downloading the model `all-minilm:latest` on a machine running Ubuntu 22.04.5 LTS (GNU/Linux 6.8.0-60-generic x86_64) with Ollama installed as a service. Hence, the command `sudo .venv/bin/od model-download all-minilm` was used.

_Notice that there are warnings that SSL verification has been disabled. This is intentional to illustrate the process of downloading through a HTTPS proxy (picked up from the `HTTPS_PROXY` environment variable) that has self-signed certificates_.

![demo-model-download](https://raw.githubusercontent.com/anirbanbasu/ollama-downloader/master/screencasts/demo_model_download.gif "model-download demo")

### `hf-model-download`

The `hfmodel-download` downloads the specified model from Hugging Face.

During the process of downloading, the following are performed.

1. Validation of the manifest for the specified model for the specified repository and organisation. _Note that not all Hugging Face models have the necessary files that can be downloaded into Ollama automatically._
2. Validation of the SHA256 hash of each downloaded BLOB.
3. Post-download verification with the Ollama server specified by `ollama_server.url` in the configuration that the downloaded model is available.

As an example, run `uv run od model-download unsloth/gemma-3-270m-it-GGUF:Q4_K_M` to download the `gemma-3-270m-it-GGUF:Q4_K_M` model from `unsloth`, the details of which can be found at https://huggingface.co/unsloth/gemma-3-270m-it-GGUF.

Running `uv run od hf-model-download --help` displays the following.

```bash
Usage: od hf-model-download [OPTIONS] ORG_REPO_MODEL

 Downloads a specified Hugging Face model.


╭─ Arguments ────────────────────────────────────────────────────────────────────╮
│ *    org_repo_model      TEXT  The name of the specific Hugging Face model to  │
│                                download, specified as <org>/<repo>:<model>,    │
│                                e.g.,                                           │
│                                bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M.    │
│                                [default: None]                                 │
│                                [required]                                      │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                    │
╰────────────────────────────────────────────────────────────────────────────────╯
```

## Testing and coverage

To run the provided set of tests using `pytest`, execute the following in _WD_. Append the flag `--capture=tee-sys` to the following command to see the console output during the tests. Note that the model download tests run as sub-processes. Their outputs will not be visible by using this flag.

```bash
uv run --group test pytest tests/
```

To get a report on coverage while invoking the tests, run the following two commands.

```bash
uv run --group test coverage run -m pytest tests/
uv run coverage report
```

## Contributing

Install [`pre-commit`](https://pre-commit.com/) for Git by using the `--all-groups` flag for `uv sync`.

Then enable `pre-commit` by running the following in the _WD_.

```bash
pre-commit install
```
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/).
