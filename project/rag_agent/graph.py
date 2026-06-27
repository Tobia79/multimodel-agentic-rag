from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

import config

from .graph_state import State, AgentState
from .nodes import (
    aggregate_answers,
    collect_answer,
    compress_context,
    direct_answer,
    evaluate_retrieval_confidence,
    fallback_response,
    orchestrator,
    request_clarification,
    rewrite_query,
    route_query,
    should_compress_context,
    summarize_history,
)
from .edges import route_after_classify, route_after_orchestrator_call, route_after_rewrite

def create_agent_graph(llm, tools_list, tool_factory=None, kb_meta_provider=None):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    if tool_factory is not None and config.CONFIDENCE_ENABLED:
        agent_builder.add_node(
            "evaluate_retrieval_confidence",
            partial(evaluate_retrieval_confidence, tool_factory=tool_factory),
        )
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(should_compress_context)
    agent_builder.add_node(collect_answer)

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    if tool_factory is not None and config.CONFIDENCE_ENABLED:
        agent_builder.add_edge("tools", "evaluate_retrieval_confidence")
    else:
        agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

    agent_subgraph = agent_builder.compile()

    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

    if config.QUERY_ROUTING_ENABLED:
        graph_builder.add_node(
            "route_query",
            partial(route_query, llm=llm, kb_meta_provider=kb_meta_provider),
        )
        graph_builder.add_node("direct_answer", partial(direct_answer, llm=llm, kb_meta_provider=kb_meta_provider))

        graph_builder.add_edge(START, "summarize_history")
        graph_builder.add_edge("summarize_history", "route_query")
        graph_builder.add_conditional_edges(
            "route_query",
            route_after_classify,
            {
                "direct_answer": "direct_answer",
                "rewrite_query": "rewrite_query",
                "request_clarification": "request_clarification",
            },
        )
        graph_builder.add_edge("request_clarification", "route_query")
        graph_builder.add_edge("direct_answer", END)
    else:
        graph_builder.add_edge(START, "summarize_history")
        graph_builder.add_edge("summarize_history", "rewrite_query")
        graph_builder.add_edge("request_clarification", "rewrite_query")

    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    print("[OK] Agent graph compiled successfully.")
    return agent_graph
