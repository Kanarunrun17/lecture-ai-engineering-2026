"""第5回 宿題（実装）: BM25 とハイブリッド検索のテスト。APIは呼ばない。"""

import json

import pytest
from pydantic_ai import Tool
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import retrieval, tools
from app.agent import agent


def test_tokenize_keeps_ascii_words_and_splits_japanese_into_bigrams():
    assert retrieval.tokenize("ＬｏＲＡ 固定する") == ["lora", "固定", "定す", "する"]
    assert retrieval.tokenize("GPU1台で650億") == ["gpu1", "台で", "650", "億"]
    assert retrieval.tokenize("MEMORY.md") == ["memory", "md"]


def test_ascii_terms_do_not_match_inside_other_words():
    # "RSI" が全ページのフッター "UNIVERSITY" に部分一致しないこと
    hits = retrieval.keyword_search("RSI", limit=300)
    assert 0 < len(hits) < 20


def test_keyword_search_finds_exact_terms_in_real_materials():
    # 実際の教材で、固有名詞と数値の一致から根拠ページ（第2回 p.52）が1位になる
    hits = retrieval.keyword_search("QLoRA 48GB")
    assert hits[0]["page_id"] == "lec02-main-p052"
    assert all(hit["bm25"] > 0 for hit in hits)


def test_keyword_search_rejects_empty_query():
    with pytest.raises(ValueError):
        retrieval.keyword_search("  ")


def test_reciprocal_rank_fusion_rewards_pages_found_by_both():
    scores = retrieval.reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["b"] > scores["a"] > scores["c"]


def test_hybrid_search_keeps_pages_found_only_by_keywords(monkeypatch):
    page = lambda page_id: {"page_id": page_id, "lecture": 2, "page": 1, "title": page_id, "content": ""}
    monkeypatch.setattr(retrieval, "search", lambda query, limit: [
        {**page("v1"), "distance": 0.1}, {**page("both"), "distance": 0.2}])
    monkeypatch.setattr(retrieval, "keyword_search", lambda query, limit: [
        {**page("both"), "bm25": 3.0}, {**page("k1"), "bm25": 1.0}])

    hits = retrieval.hybrid_search("質問")

    assert [hit["page_id"] for hit in hits] == ["both", "v1", "k1"]
    assert (hits[0]["vector_rank"], hits[0]["keyword_rank"]) == (2, 1)
    assert hits[2]["vector_rank"] is None  # ベクトル検索では圏外だったページも残る
    assert "distance" not in hits[0] and "bm25" not in hits[0]


def test_hybrid_tool_looks_the_same_to_the_model(monkeypatch):
    monkeypatch.setattr("app.retrieval.hybrid_search", lambda query: [{"page_id": "p", "rrf_score": 0.03}])
    model = TestModel(call_tools=["search_materials"])
    hybrid = Tool(tools.search_materials_hybrid, name="search_materials")
    with agent.override(model=model, tools=[hybrid]):
        result = agent.run_sync("LoRAとは？")
    offered = model.last_model_request_parameters.function_tools
    assert [t.name for t in offered] == ["search_materials"]
    assert offered[0].description.startswith(tools.search_materials.__doc__.splitlines()[0])
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert json.loads(returns[0].content)["results"][0]["rrf_score"] == 0.03
