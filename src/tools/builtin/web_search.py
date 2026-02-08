import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.tools.base import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT_SECONDS = 10
MAX_RESULTS = 5


class WebSearchTool:
    """
    Performs a web search and returns summarized search results.
    Uses DuckDuckGo HTML endpoint (no API key required).
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description=(
                "Search the web for information based on a query and "
                "return top results with titles and links."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to find information on the web.",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs.get("query", "").strip()

        if not query:
            return "Error: 'query' parameter is required."

        logger.info(f"Performing web search for query: {query}")

        search_url = "https://duckduckgo.com/html/"
        params = {"q": query}

        try:
            async with httpx.AsyncClient(
                timeout=SEARCH_TIMEOUT_SECONDS, follow_redirects=True
            ) as client:
                response = await client.get(search_url, params=params)

            if not response.is_success:
                return f"Error: Search request failed with status code {response.status_code}."

            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.select(".result"):
                title_tag = result.select_one(".result__a")
                snippet_tag = result.select_one(".result__snippet")

                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                link = title_tag.get("href", "")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                results.append((title, link, snippet))

                if len(results) >= MAX_RESULTS:
                    break

            if not results:
                return f"No search results found for query: '{query}'."

            formatted_results = []
            for idx, (title, link, snippet) in enumerate(results, start=1):
                formatted_results.append(
                    f"[{idx}] {title}\n{link}\n{snippet}"
                )

            return (
                f"Web search results for '{query}':\n\n"
                + "\n\n".join(formatted_results)
            )

        except httpx.RequestError as e:
            logger.exception("Search request failed")
            return f"Error: Failed to perform web search: {str(e)}"
        except Exception as e:
            logger.exception("Unexpected search error")
            return f"Error: Unexpected failure during web search: {str(e)}"
