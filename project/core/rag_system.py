import uuid
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from rag_agent.query_router import make_kb_meta_provider
from core.observability import Observability


def create_llm():
    if config.LLM_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI

        if not config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.LLM_BASE_URL,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)


class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        self.agent_graph = None
        self.tool_factory = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = create_llm()
        self.tool_factory = ToolFactory(collection, llm=llm)
        tools = self.tool_factory.create_tools()
        kb_meta_provider = make_kb_meta_provider(self)
        self.agent_graph = create_agent_graph(
            llm,
            tools,
            tool_factory=self.tool_factory,
            kb_meta_provider=kb_meta_provider,
        )

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())