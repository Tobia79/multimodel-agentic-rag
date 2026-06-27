"""Query routing: direct LLM answer vs RAG retrieval."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

import config
from rag_agent.prompts import get_route_query_prompt
from rag_agent.schemas import RouteDecision

logger = logging.getLogger(__name__)

RouteType = Literal["direct", "rag", "clarify"]

DIRECT_PATTERNS = (
    r"^(你好|hi|hello|hey|谢谢|感谢|多谢|再见|拜拜|早上好|晚上好)\b",
    r"(写|生成|实现|编写).{0,12}(代码|脚本|函数|程序|类)",
    r"^(翻译|translate)\b",
    r"^(用|使用)?(python|javascript|java|c\+\+|rust|go)\s*(写|实现|解释)",
)

# Meta questions about KB inventory — direct (list docs), NOT content retrieval
KB_INVENTORY_PATTERNS = (
    r"知识库.*(有什么|有哪些|有什么资料|有什么文档|有什么文件|装了什么|包含什么|能查什么|可以查什么)",
    r"(有什么|有哪些).{0,12}(资料|文档|文件).{0,12}(知识库|上传|入库|可以查|能查)",
    r"(目前|现在|当前|我).{0,8}(知识库|文档库).{0,20}(有什么|有哪些|列表|清单|可以查|能查)",
    r"(上传|导入)了哪些",
    r"^(列出|列表|清单|显示).{0,8}(文档|资料|知识库|文件)",
    r"what.{0,20}(documents?|files?).{0,20}(knowledge base|kb|uploaded)",
    r"(list|show).{0,12}(documents?|files?).{0,12}(knowledge base|uploaded|kb)",
)

RAG_PATTERNS = (
    r"(文档|知识库|资料|手册|制度|规范|报告|论文)",
    r"(根据|依据|引用|出处|原文|第[\d一二三四五六七八九十]+[章节条款页])",
    r"(上传|导入)的",
    r"(总结|概括|归纳).{0,8}(文档|资料|知识库)",
)

FOLLOW_UP_MARKERS = ("它", "这个", "那个", "呢", "还有吗", "然后呢", "继续", "上面", "刚才")

SOURCE_LABELS = {
    "rule": "规则",
    "llm": "LLM",
    "override": "用户强制",
    "empty_kb": "空知识库",
    "threshold_fallback": "置信度兜底",
    "disabled": "路由已关闭",
    "rules_only_default": "规则未命中默认",
}

ROUTE_LABELS = {
    "direct": "Direct 直答",
    "rag": "RAG 检索",
    "clarify": "需要澄清",
}


@dataclass
class KBMeta:
    doc_count: int = 0
    doc_names: list[str] = field(default_factory=list)
    doc_titles: list[str] = field(default_factory=list)


@dataclass
class ResolvedRouteDecision:
    route: RouteType
    source: str
    confidence: float
    reason: str
    clarification_needed: str = ""


_ROUTE_JSON_HINT = (
    "\n\nRespond with a single JSON object only, using exactly these keys:\n"
    '- "route": "direct" | "rag" | "clarify"\n'
    '- "confidence": number between 0 and 1\n'
    '- "reason": string (brief, max 200 chars)\n'
    '- "clarification_needed": string (empty unless route is clarify)\n'
)


def format_route_display(route: str, source: str, confidence: float, reason: str) -> str:
    route_label = ROUTE_LABELS.get(route, route)
    source_label = SOURCE_LABELS.get(source, source)
    return (
        f"✅ **路由：{route_label}**\n"
        f"来源：{source_label} | 置信度：{confidence:.2f}\n"
        f"原因：{reason}"
    )


def _normalize_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        cleaned = (name or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _mentions_any_doc(query_lower: str, kb_meta: KBMeta) -> bool:
    for name in kb_meta.doc_names:
        if name.lower() in query_lower:
            return True
    return False


def _is_follow_up(query: str) -> bool:
    stripped = query.strip()
    if len(stripped) > 30:
        return False
    return any(marker in stripped for marker in FOLLOW_UP_MARKERS)


def _summary_mentions_docs(summary: str, kb_meta: KBMeta) -> bool:
    summary_lower = summary.lower()
    return any(name.lower() in summary_lower for name in kb_meta.doc_names)


def _is_kb_inventory_query(query: str) -> bool:
    q_lower = query.strip().lower()
    return any(re.search(pattern, q_lower) for pattern in KB_INVENTORY_PATTERNS)


def format_kb_inventory(kb_meta: KBMeta) -> str:
    if kb_meta.doc_count == 0:
        return "当前知识库为空，尚未上传任何文档。"
    lines = [f"共 {kb_meta.doc_count} 份文档："]
    for index, title in enumerate(kb_meta.doc_titles[:30], start=1):
        lines.append(f"{index}. {title}")
    if kb_meta.doc_count > 30:
        lines.append(f"... 另有 {kb_meta.doc_count - 30} 份未列出")
    return "\n".join(lines)


def apply_overrides(
    *,
    force_rag: bool,
    kb_meta: KBMeta,
) -> Optional[ResolvedRouteDecision]:
    if force_rag:
        return ResolvedRouteDecision(
            route="rag",
            source="override",
            confidence=1.0,
            reason="用户勾选「强制查知识库」",
        )

    if not config.QUERY_ROUTING_ENABLED:
        return ResolvedRouteDecision(
            route="rag",
            source="disabled",
            confidence=1.0,
            reason="查询路由功能已关闭，默认走 RAG",
        )

    if kb_meta.doc_count == 0:
        return ResolvedRouteDecision(
            route="direct",
            source="empty_kb",
            confidence=1.0,
            reason="知识库为空，使用直答模式",
        )

    return None


def rule_route(query: str, kb_meta: KBMeta, summary: str) -> Optional[ResolvedRouteDecision]:
    q = query.strip()
    if not q:
        return ResolvedRouteDecision(
            route="clarify",
            source="rule",
            confidence=1.0,
            reason="输入为空",
            clarification_needed="请输入您的问题。",
        )

    q_lower = q.lower()

    if _is_kb_inventory_query(q):
        return ResolvedRouteDecision(
            route="direct",
            source="rule",
            confidence=0.98,
            reason="询问知识库清单（元查询），无需检索正文",
        )

    for pattern in RAG_PATTERNS:
        if re.search(pattern, q_lower):
            return ResolvedRouteDecision(
                route="rag",
                source="rule",
                confidence=0.95,
                reason="命中 RAG 关键词",
            )

    if _mentions_any_doc(q_lower, kb_meta):
        return ResolvedRouteDecision(
            route="rag",
            source="rule",
            confidence=0.98,
            reason="提及知识库文档名称",
        )

    if summary and _is_follow_up(q) and _summary_mentions_docs(summary, kb_meta):
        return ResolvedRouteDecision(
            route="rag",
            source="rule",
            confidence=0.85,
            reason="文档上下文下的追问",
        )

    for pattern in DIRECT_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            if not _mentions_any_doc(q_lower, kb_meta):
                return ResolvedRouteDecision(
                    route="direct",
                    source="rule",
                    confidence=0.90,
                    reason="通用/闲聊类问题",
                )

    return None


def finalize_route(decision: RouteDecision, kb_meta: KBMeta) -> ResolvedRouteDecision:
    route = decision.route
    clarification = (decision.clarification_needed or "").strip()

    if route == "clarify" and not clarification:
        route = "rag"
        clarification = ""

    resolved = ResolvedRouteDecision(
        route=route,
        source="llm",
        confidence=decision.confidence,
        reason=decision.reason.strip() or "LLM 路由判定",
        clarification_needed=clarification,
    )

    if resolved.route == "direct" and kb_meta.doc_count > 0:
        if resolved.confidence < config.QUERY_ROUTING_LLM_THRESHOLD:
            return ResolvedRouteDecision(
                route="rag",
                source="threshold_fallback",
                confidence=resolved.confidence,
                reason=(
                    f"LLM 直答置信度 {resolved.confidence:.2f} 低于阈值 "
                    f"{config.QUERY_ROUTING_LLM_THRESHOLD}，降级为 RAG"
                ),
            )

    return resolved


def _run_llm_route(llm, context_section: str, kb_meta: KBMeta) -> ResolvedRouteDecision:
    configured = llm.with_config(temperature=0.0)
    system_prompt = get_route_query_prompt(kb_meta)

    if config.LLM_PROVIDER == "deepseek":
        response = configured.bind(response_format={"type": "json_object"}).invoke(
            [
                SystemMessage(content=system_prompt + _ROUTE_JSON_HINT),
                HumanMessage(content=context_section),
            ]
        )
        decision = RouteDecision.model_validate_json(response.content)
    else:
        decision = configured.with_structured_output(RouteDecision).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context_section),
            ]
        )

    return finalize_route(decision, kb_meta)


def llm_route(
    llm,
    query: str,
    summary: str,
    kb_meta: KBMeta,
) -> ResolvedRouteDecision:
    context_parts = []
    if summary.strip():
        context_parts.append(f"Conversation Context:\n{summary.strip()}")
    context_parts.append(f"User Query:\n{query.strip()}")
    context_section = "\n\n".join(context_parts)

    try:
        return _run_llm_route(llm, context_section, kb_meta)
    except Exception as exc:
        logger.warning("LLM route failed, defaulting to RAG: %s", exc)
        return ResolvedRouteDecision(
            route="rag",
            source="threshold_fallback",
            confidence=0.0,
            reason=f"LLM 路由失败，默认 RAG：{exc}",
        )


def resolve_route(
    *,
    query: str,
    summary: str,
    kb_meta: KBMeta,
    force_rag: bool = False,
    llm=None,
) -> ResolvedRouteDecision:
    override = apply_overrides(force_rag=force_rag, kb_meta=kb_meta)
    if override is not None:
        return override

    if config.QUERY_ROUTING_USE_RULES:
        ruled = rule_route(query, kb_meta, summary)
        if ruled is not None:
            return ruled

    if config.QUERY_ROUTING_RULES_ONLY:
        return ResolvedRouteDecision(
            route="rag",
            source="rules_only_default",
            confidence=0.75,
            reason="规则未命中，默认 RAG（RULES_ONLY 模式）",
        )

    if llm is None:
        return ResolvedRouteDecision(
            route="rag",
            source="threshold_fallback",
            confidence=0.0,
            reason="无 LLM 实例，默认 RAG",
        )

    return llm_route(llm, query, summary, kb_meta)


def build_kb_meta_from_docs(docs) -> KBMeta:
    """Build KBMeta from DocumentManager.list_documents() results."""
    names: list[str] = []
    titles: list[str] = []
    for doc in docs:
        if doc.display_name:
            names.append(doc.display_name)
        if doc.stem:
            names.append(doc.stem)
        if doc.source_name:
            names.append(doc.source_name)
        titles.append(doc.title or doc.display_name or doc.stem or doc.source_name)

    return KBMeta(
        doc_count=len(docs),
        doc_names=_normalize_names(names),
        doc_titles=_normalize_names(titles),
    )


def make_kb_meta_provider(rag_system) -> Callable[[], KBMeta]:
    """Return a callable that reads the current knowledge base on each invocation."""

    def provider() -> KBMeta:
        from core.document_manager import DocumentManager

        try:
            docs = DocumentManager(rag_system).list_documents(fast=True)
            return build_kb_meta_from_docs(docs)
        except Exception as exc:
            logger.warning("Failed to load KB metadata for routing: %s", exc)
            return KBMeta()

    return provider
