from typing import Literal, Set
import logging
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from rag_agent.query_router import format_kb_inventory, resolve_route
from utils import estimate_context_tokens
import config
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR

logger = logging.getLogger(__name__)

_QUERY_ANALYSIS_JSON_HINT = (
    "\n\nRespond with a single JSON object only, using exactly these keys:\n"
    '- "is_clear": boolean\n'
    '- "questions": array of strings (rewritten queries when clear)\n'
    '- "clarification_needed": string (empty when not needed)\n'
)


def _run_query_analysis(llm, context_section: str) -> QueryAnalysis:
    configured = llm.with_config(temperature=0.1)
    system_prompt = get_rewrite_query_prompt()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=context_section)]

    if config.LLM_PROVIDER == "deepseek":
        response = configured.bind(response_format={"type": "json_object"}).invoke(
            [
                SystemMessage(content=system_prompt + _QUERY_ANALYSIS_JSON_HINT),
                HumanMessage(content=context_section),
            ]
        )
        return QueryAnalysis.model_validate_json(response.content)

    return configured.with_structured_output(QueryAnalysis).invoke(messages)

def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def route_query(state: State, llm, kb_meta_provider):
    last_message = state["messages"][-1]
    query = last_message.content
    summary = state.get("conversation_summary", "")

    kb_meta = kb_meta_provider() if kb_meta_provider is not None else None
    if kb_meta is None:
        from rag_agent.query_router import KBMeta
        kb_meta = KBMeta()

    decision = resolve_route(
        query=query,
        summary=summary,
        kb_meta=kb_meta,
        force_rag=state.get("force_rag", False),
        llm=llm,
    )

    update = {
        "query_route": decision.route,
        "route_source": decision.source,
        "route_confidence": decision.confidence,
        "route_reason": decision.reason,
        "originalQuery": query,
    }

    if decision.route == "clarify":
        clarification = decision.clarification_needed or "请补充更多细节，以便我判断是否需要查阅知识库。"
        update["questionIsClear"] = False
        update["messages"] = [AIMessage(content=clarification)]
        return update

    update["questionIsClear"] = True
    return update

def direct_answer(state: State, llm, kb_meta_provider=None):
    summary = state.get("conversation_summary", "").strip()
    query = state.get("originalQuery") or state["messages"][-1].content

    kb_meta = kb_meta_provider() if kb_meta_provider is not None else None
    if kb_meta is None:
        from rag_agent.query_router import KBMeta
        kb_meta = KBMeta()

    parts = []
    if summary:
        parts.append(f"对话摘要：\n{summary}")
    parts.append(f"当前知识库文档清单：\n{format_kb_inventory(kb_meta)}")
    parts.append(f"用户问题：\n{query}")
    user_content = "\n\n".join(parts)

    response = llm.with_config(temperature=0.3).invoke([
        SystemMessage(content=get_direct_answer_prompt()),
        HumanMessage(content=user_content),
    ])
    return {"messages": [response]}

def rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    response = _run_query_analysis(llm, context_section)

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def evaluate_retrieval_confidence(state: AgentState, tool_factory):
    """Route after search based on layered retrieval confidence (CRAG-style tiers)."""
    outcome = getattr(tool_factory, "last_retrieval_outcome", None)
    tool_factory.last_retrieval_outcome = None

    if outcome is None or not outcome.from_search:
        return Command(goto="should_compress_context")

    update = {
        "retrieval_confidence": outcome.confidence_score,
        "retrieval_confidence_tier": outcome.tier,
    }

    if outcome.tier == "medium" and config.CONFIDENCE_AGENT_RETRY_ON_MEDIUM:
        retries = state.get("confidence_retry_count", 0)
        if retries < config.CONFIDENCE_MAX_AGENT_RETRIES:
            reason = outcome.reasoning or "Retrieved excerpts may be incomplete for the question."
            hint = HumanMessage(
                content=(
                    f"[RETRIEVAL CONFIDENCE {outcome.confidence_score:.0f}/10 — MEDIUM]\n"
                    f"{reason}\n"
                    "Broaden or rephrase the search query, then call 'search_child_chunks' again "
                    "for missing aspects only."
                )
            )
            return Command(
                update={**update, "confidence_retry_count": retries + 1, "messages": [hint]},
                goto="orchestrator",
            )

    if outcome.tier == "low":
        if config.WEB_SEARCH_ENABLED and config.CONFIDENCE_WEB_SEARCH_ON_LOW:
            web_retries = state.get("web_search_retry_count", 0)
            if web_retries < config.CONFIDENCE_MAX_WEB_SEARCH_RETRIES:
                reason = outcome.reasoning or "Local knowledge base may not cover this question."
                hint = HumanMessage(
                    content=(
                        f"[RETRIEVAL CONFIDENCE {outcome.confidence_score:.0f}/10 — LOW]\n"
                        f"{reason}\n"
                        "Local documents appear insufficient. Call 'web_search' with a focused query "
                        "to fetch external information, then synthesize an answer using both sources."
                    )
                )
                return Command(
                    update={**update, "web_search_retry_count": web_retries + 1, "messages": [hint]},
                    goto="orchestrator",
                )

        logger.info(
            "Low retrieval confidence (%.1f/10); continuing without web search fallback.",
            outcome.confidence_score,
        )

    return Command(update=update, goto="should_compress_context")


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")

                elif tc["name"] == "web_search":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"web::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))
        web_queries = sorted(r.replace("web::", "") for r in retrieved_ids if r.startswith("web::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        if web_queries:
            block += "Web searches already run:\n" + "\n".join(f"- {q}" for q in web_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}