"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
import unicodedata
from collections import Counter, defaultdict, deque
from functools import cache
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    HYBRID_CANDIDATES, RRF_K, BM25_K1, BM25_B,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str, limit: int = RAG_TOP_K) -> list[dict]:
    """質問を埋め込み、近いページを最大limit件（既定はRAG_TOP_K件）返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), limit))
        return [dict(row) for row in rows]


# ---- 第5回 宿題（実装）: キーワード検索（BM25）とハイブリッド検索 ----
#
# ベクトル検索は「意味が近いページ」を探すが、「QLoRA」「4bit」「48GB」のような
# 固有名詞・数値の一致は苦手なことがある。そこで語の一致で探す BM25 を併用し、
# 2つの順位を RRF でまとめる。
#
# 日本語は単語の区切りに空白がないため、形態素解析（Janome など）の代わりに
# 文字 bigram（2文字ずつ区切る）を語として扱う。追加パッケージが不要で、
# 「固定」「学習」のような2文字の語も拾える。SQLite FTS5 の trigram は
# 3文字未満の語を検索できないため採用しなかった。
#
# 英数字は bigram にせず、単語のまま1語にする。最初は英数字も bigram にしていたが、
# 「RSI」が全ページのフッター「UNIVERSITY」の中の "rsi" に一致してしまい、
# 214ページすべてがヒットしたため分けた。

def tokenize(text: str) -> list[str]:
    """表記を揃え（全角英数→半角・小文字）、英数字は単語、それ以外は文字 bigram に分ける。"""
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for word in re.split(r"[\s\W_]+", normalized):
        for segment in re.findall(r"[a-z0-9]+|[^a-z0-9]+", word):
            if segment.isascii() or len(segment) == 1:
                tokens.append(segment)  # 英数字の単語と、1文字だけの語（「量」など）はそのまま
            else:
                tokens.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return tokens


@cache
def _bm25_index() -> tuple[list[dict], list[Counter], dict[str, int], float]:
    """対象ページの bigram 頻度を一度だけ数えて保持する（214ページなので一瞬で終わる）。"""
    pages = load_pages(RAG_SOURCE)
    term_counts = [Counter(tokenize(f"{page['title']}\n{page['content']}")) for page in pages]
    document_frequency = Counter(term for counts in term_counts for term in counts)
    average_length = sum(sum(counts.values()) for counts in term_counts) / max(len(pages), 1)
    return pages, term_counts, dict(document_frequency), average_length


def keyword_search(query: str, limit: int = RAG_TOP_K) -> list[dict]:
    """BM25 で語の一致が多いページを、スコアの高い順に最大limit件返す。APIは呼ばない。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    pages, term_counts, document_frequency, average_length = _bm25_index()
    query_terms = set(tokenize(query))
    total = len(pages)
    scored = []
    for page, counts in zip(pages, term_counts):
        length = sum(counts.values())
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log((total - df + 0.5) / (df + 0.5) + 1)
            score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / average_length))
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda item: (-item[0], item[1]["page_id"]))
    return [{**page, "bm25": round(score, 4)} for score, page in scored[:limit]]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """複数の順位リストを RRF でまとめる。各リストで r 位なら 1/(k+r) 点を足す。

    スコアの尺度（コサイン距離と BM25）が違っても、順位だけを使うので混ぜられる。
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, 1):
            scores[item_id] += 1 / (k + rank)
    return dict(scores)


def format_rank(rank: int | None) -> str:
    """順位を表示用にする。候補に入らなかったときは「圏外」。"""
    return f"{rank}位" if rank else "圏外"


def hybrid_search(query: str) -> list[dict]:
    """ベクトル検索と BM25 の上位候補を RRF でまとめ、最大RAG_TOP_K件返す。

    各結果には、それぞれの検索での順位（vector_rank / keyword_rank、圏外なら None）と
    RRF スコアを付け、どちらの検索で拾えたページかを確認できるようにする。
    """
    vector_hits = search(query, limit=HYBRID_CANDIDATES)
    keyword_hits = keyword_search(query, limit=HYBRID_CANDIDATES)
    vector_rank = {hit["page_id"]: rank for rank, hit in enumerate(vector_hits, 1)}
    keyword_rank = {hit["page_id"]: rank for rank, hit in enumerate(keyword_hits, 1)}
    hits = {hit["page_id"]: hit for hit in keyword_hits} | {hit["page_id"]: hit for hit in vector_hits}
    fused = reciprocal_rank_fusion([list(vector_rank), list(keyword_rank)])
    ordered = sorted(fused, key=lambda page_id: (-fused[page_id], page_id))[:RAG_TOP_K]
    results = []
    for page_id in ordered:
        hit = {key: value for key, value in hits[page_id].items() if key not in ("distance", "bm25")}
        results.append({**hit, "rrf_score": round(fused[page_id], 5),
                        "vector_rank": vector_rank.get(page_id), "keyword_rank": keyword_rank.get(page_id)})
    return results


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--keyword", action="store_true", help="第5回 宿題: BM25（語の一致）だけで検索する。APIは呼ばない")
    method.add_argument("--hybrid", action="store_true", help="第5回 宿題: ベクトル検索とBM25をRRFでまとめる")
    parser.add_argument("--brief", action="store_true", help="本文を省き、順位と見出しだけを表示する")
    args = parser.parse_args()
    finder = keyword_search if args.keyword else hybrid_search if args.hybrid else search
    for i, result in enumerate(finder(args.query), 1):
        if args.brief:
            ranks = ""
            if "rrf_score" in result:
                ranks = f" [ベクトル {format_rank(result['vector_rank'])} / BM25 {format_rank(result['keyword_rank'])}]"
            print(f"{i}. 第{result['lecture']}回 p.{result['page']} {result['title']}{ranks}")
            continue
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        if "rrf_score" in result:
            print(f"順位: ベクトル {format_rank(result['vector_rank'])} / BM25 {format_rank(result['keyword_rank'])}"
                  f"（RRF {result['rrf_score']}）")
        print(f"\n{result['content']}")
