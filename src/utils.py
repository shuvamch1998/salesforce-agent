import time
from langchain_community.tools import DuckDuckGoSearchRun

class ThrottledDuckDuckGoSearch(DuckDuckGoSearchRun):
    def _run(self, query: str) -> str:
        time.sleep(5)  
        return super()._run(query)
