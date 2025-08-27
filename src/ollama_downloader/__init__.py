import logging
from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, markup=True, show_path=False)],
)
