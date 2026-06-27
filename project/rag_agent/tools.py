from langchain_core.tools import tool
import config
from core.retrieval import retrieve_with_confidence
from core.web_search import search_web
from db.parent_store_manager import ParentStoreManager
from rag_agent.chunk_format import format_parent_chunk_for_agent

class ToolFactory:
    
    def __init__(self, collection, llm=None):
        self.collection = collection
        self.llm = llm
        self.parent_store_manager = ParentStoreManager()
        self.last_retrieval_outcome = None
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return (capped at FUSION_TOP_K)
        """
        try:
            effective_limit = min(max(1, int(limit)), config.FUSION_TOP_K)
            outcome = retrieve_with_confidence(
                self.collection,
                query,
                effective_limit,
                llm=self.llm if config.CONFIDENCE_LLM_ENABLED else None,
            )
            self.last_retrieval_outcome = outcome
            results = outcome.documents
            if not results:
                return "NO_RELEVANT_CHUNKS"

            footer = (
                f"\n\n[retrieval_confidence={outcome.confidence_score:.1f}/10;"
                f" tier={outcome.tier}; source={outcome.confidence_source}]"
            )
            if outcome.secondary_retrieval_used:
                footer = footer[:-1] + "; secondary_retrieval=true]"

            body = "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
            return body + footer

        except Exception as e:
            self.last_retrieval_outcome = None
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        self.last_retrieval_outcome = None
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return format_parent_chunk_for_agent(
                parent_id,
                parent.get("content", ""),
                parent.get("metadata"),
            )

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """Search the public web when local documents are insufficient.

        Args:
            query: Focused web search query
            max_results: Maximum number of results to return (capped by WEB_SEARCH_MAX_RESULTS)
        """
        self.last_retrieval_outcome = None
        if not config.WEB_SEARCH_ENABLED:
            return "WEB_SEARCH_DISABLED"
        try:
            effective_limit = min(max(1, int(max_results)), config.WEB_SEARCH_MAX_RESULTS)
            return search_web(query, max_results=effective_limit)
        except Exception as e:
            return f"WEB_SEARCH_ERROR: {str(e)}"

    def create_tools(self) -> list:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        tools = [search_tool, retrieve_tool]

        if config.WEB_SEARCH_ENABLED:
            web_tool = tool("web_search")(self._web_search)
            tools.append(web_tool)

        return tools
