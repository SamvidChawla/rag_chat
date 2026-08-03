import logging
import sys

from app.config import settings


def setup_logging() -> None:
    level = logging.DEBUG if settings.app_env == "dev" else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quiet noisy third-party loggers unless we're actively debugging them
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)