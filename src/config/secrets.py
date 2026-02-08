import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Secrets:
    brave_search_api_key: str = ""

    def has_brave_search(self) -> bool:
        return bool(self.brave_search_api_key)


def load_secrets(env_path: Path = Path(".env")) -> Secrets:
    """Load secrets from environment variables and .env file."""
    load_dotenv(env_path)

    return Secrets(
        brave_search_api_key=os.environ.get("BRAVE_SEARCH_API_KEY", ""),
    )