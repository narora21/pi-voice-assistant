import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Secrets:
    brave_search_api_key: str = ""
    kasa_username: str = ""
    kasa_password: str = ""

    def has_brave_search(self) -> bool:
        return bool(self.brave_search_api_key)

    def has_kasa_credentials(self) -> bool:
        return bool(self.kasa_username and self.kasa_password)


def load_secrets(env_path: Path = Path(".env")) -> Secrets:
    """Load secrets from environment variables and .env file."""
    load_dotenv(env_path)

    return Secrets(
        brave_search_api_key=os.environ.get("BRAVE_SEARCH_API_KEY", ""),
        kasa_username=os.environ.get("KASA_USERNAME", ""),
        kasa_password=os.environ.get("KASA_PASSWORD", ""),
    )