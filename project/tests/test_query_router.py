"""Tests for query routing (direct vs RAG)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from rag_agent.query_router import (
    KBMeta,
    apply_overrides,
    finalize_route,
    resolve_route,
    rule_route,
)
from rag_agent.schemas import RouteDecision


def _kb_with_docs():
    return KBMeta(
        doc_count=2,
        doc_names=["集合01", "集合01.md", "manual.pdf"],
        doc_titles=["集合01", "Employee Manual"],
    )


def test_empty_kb_routes_direct():
    decision = apply_overrides(force_rag=False, kb_meta=KBMeta())
    assert decision is not None
    assert decision.route == "direct"
    assert decision.source == "empty_kb"


def test_force_rag_override():
    decision = apply_overrides(force_rag=True, kb_meta=KBMeta())
    assert decision is not None
    assert decision.route == "rag"
    assert decision.source == "override"


def test_rule_kb_inventory_query():
    decision = rule_route("我目前的知识库里有什么资料可以查", _kb_with_docs(), "")
    assert decision is not None
    assert decision.route == "direct"
    assert decision.source == "rule"


def test_rule_rag_keyword():
    decision = rule_route("根据知识库总结安全规范", _kb_with_docs(), "")
    assert decision is not None
    assert decision.route == "rag"


def test_rule_rag_doc_name():
    decision = rule_route("集合01 第三章讲了什么？", _kb_with_docs(), "")
    assert decision is not None
    assert decision.route == "rag"
    assert decision.source == "rule"


def test_rule_direct_greeting():
    decision = rule_route("你好", _kb_with_docs(), "")
    assert decision is not None
    assert decision.route == "direct"


def test_rule_follow_up_with_doc_context():
    summary = "User asked about 集合01 document structure."
    decision = rule_route("它呢？", _kb_with_docs(), summary)
    assert decision is not None
    assert decision.route == "rag"


def test_finalize_low_confidence_direct_downgrades_to_rag():
    kb = _kb_with_docs()
    raw = RouteDecision(route="direct", confidence=0.4, reason="maybe general")
    decision = finalize_route(raw, kb)
    assert decision.route == "rag"
    assert decision.source == "threshold_fallback"


def test_finalize_invalid_clarify_becomes_rag():
    kb = _kb_with_docs()
    raw = RouteDecision(route="clarify", confidence=0.5, reason="unclear", clarification_needed="")
    decision = finalize_route(raw, kb)
    assert decision.route == "rag"


def test_resolve_route_rules_only_defaults_rag(monkeypatch):
    monkeypatch.setattr(config, "QUERY_ROUTING_ENABLED", True)
    monkeypatch.setattr(config, "QUERY_ROUTING_USE_RULES", True)
    monkeypatch.setattr(config, "QUERY_ROUTING_RULES_ONLY", True)

    decision = resolve_route(
        query="Explain the trade-offs of microservices architecture in general.",
        summary="",
        kb_meta=_kb_with_docs(),
        force_rag=False,
        llm=None,
    )
    assert decision.route == "rag"
    assert decision.source == "rules_only_default"


def test_resolve_route_disabled_goes_rag(monkeypatch):
    monkeypatch.setattr(config, "QUERY_ROUTING_ENABLED", False)

    decision = resolve_route(
        query="你好",
        summary="",
        kb_meta=_kb_with_docs(),
        force_rag=False,
        llm=MagicMock(),
    )
    assert decision.route == "rag"
    assert decision.source == "disabled"
