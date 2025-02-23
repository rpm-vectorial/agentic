from langchain.tools import Tool
from typing import Optional
import json
import requests

class SearchTool():
    def __init__(self, serper_api_key: Optional[str] = None):
        self.serper_api_key = serper_api_key

    def search(self, query: str) -> str:
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json',
        }
        data = { "q": query }
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=data
        )
        return json.dumps(response.json())

    def get_tool(self) -> Tool:
        return Tool(
            name="Search",
            func=self.search,
            description="Search the internet for information about companies and people"
        )
