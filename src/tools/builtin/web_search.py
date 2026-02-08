import logging
from typing import Any

import httpx

from src.tools.base import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT_SECONDS = 10
MAX_RESULTS = 5


class WebSearchTool:
    """
    Performs a web search using Brave Search API.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description=(
                "Search the web for current information based on a query. "
                "Use this for recent events, news, or facts you're unsure about. "
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

        logger.info(f"Performing Brave web search for query: {query}")

        try:
            async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={
                        "q": query,
                        "count": MAX_RESULTS,
                    },
                    headers={
                        "X-Subscription-Token": self._api_key,
                        "Accept": "application/json",
                    },
                )

            if response.status_code == 401:
                return "Error: Invalid Brave Search API key."
            
            if response.status_code == 429:
                return "Error: Brave Search rate limit exceeded."

            if not response.is_success:
                return f"Error: Search request failed with status {response.status_code}."

            data = response.json()
            web_results = data.get("web", {}).get("results", [])

            if not web_results:
                return f"No search results found for query: '{query}'."

            formatted_results = []
            for idx, result in enumerate(web_results[:MAX_RESULTS], start=1):
                title = result.get("title", "")
                url = result.get("url", "")
                description = result.get("description", "")
                formatted_results.append(f"[{idx}] {title}\n{url}\n{description}")

            return (
                f"Web search results for '{query}':\n\n"
                + "\n\n".join(formatted_results)
            )

        except httpx.TimeoutException:
            logger.exception("Search request timed out")
            return "Error: Search request timed out."
        except httpx.RequestError as e:
            logger.exception("Search request failed")
            return f"Error: Failed to perform web search: {str(e)}"
        except Exception as e:
            logger.exception("Unexpected search error")
            return f"Error: Unexpected failure during web search: {str(e)}"
