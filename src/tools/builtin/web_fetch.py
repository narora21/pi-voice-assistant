import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.tools.base import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)

URL_FETCH_TIMEOUT_SECONDS = 10
MAX_CONTENT_LENGTH = 100_000


def _extract_urls(text: str) -> tuple[list[str], list[str]]:
    """
    Extract valid http/https URLs from text.
    Returns (valid_urls, errors)
    """
    tokens = re.split(r"\s+", text)
    valid_urls: list[str] = []
    errors: list[str] = []

    for token in tokens:
        if "://" not in token:
            continue

        try:
            parsed = urlparse(token)
            if parsed.scheme not in ("http", "https"):
                errors.append(
                    f"Unsupported protocol in URL: '{token}'. Only http and https are supported."
                )
                continue

            if not parsed.netloc:
                errors.append(f"Malformed URL detected: '{token}'.")
                continue

            valid_urls.append(token)
        except Exception:
            errors.append(f"Malformed URL detected: '{token}'.")

    return valid_urls, errors


class WebFetchTool:
    """
    Fetches and processes content from URL(s) embedded in a prompt.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_fetch",
            description=(
                "Fetch and process content from URL(s) included in a prompt. "
                "The prompt must include at least one full http:// or https:// URL "
                "and may include instructions like 'summarize' or 'extract key points'."
            ),
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description=(
                        "A prompt containing one or more full URLs (http/https) "
                        "and instructions for processing their content."
                    ),
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> str:
        prompt: str = kwargs.get("prompt", "").strip()

        if not prompt:
            return "Error: 'prompt' parameter is required."

        valid_urls, errors = _extract_urls(prompt)

        if errors:
            return "Error(s) in prompt URLs:\n- " + "\n- ".join(errors)

        if not valid_urls:
            return (
                "Error: The prompt must contain at least one valid URL "
                "(starting with http:// or https://)."
            )

        url = valid_urls[0]

        # Convert GitHub blob URLs to raw
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace(
                "/blob/", "/"
            )

        logger.info(f"Fetching URL: {url}")

        try:
            async with httpx.AsyncClient(timeout=URL_FETCH_TIMEOUT_SECONDS) as client:
                response = await client.get(url)

            if response.status_code != 200:
                return (
                    f"Error: Request failed with status code "
                    f"{response.status_code}."
                )

            content_type = response.headers.get("content-type", "")
            raw_content = response.text

            # Convert HTML to plain text if needed
            if "text/html" in content_type.lower() or not content_type:
                soup = BeautifulSoup(raw_content, "html.parser")

                # Remove scripts and styles
                for tag in soup(["script", "style"]):
                    tag.decompose()

                text_content = soup.get_text(separator="\n")
            else:
                text_content = raw_content

            text_content = text_content.strip()
            text_content = text_content[:MAX_CONTENT_LENGTH]

            return (
                f"Fetched content from {url}:\n\n"
                f"{text_content}"
            )

        except httpx.RequestError as e:
            logger.exception("HTTP request failed")
            return f"Error: Failed to fetch URL '{url}': {str(e)}"
        except Exception as e:
            logger.exception("Unexpected error during web fetch")
            return f"Error: Unexpected failure while fetching '{url}': {str(e)}"
