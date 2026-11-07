"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
from datetime import datetime
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# 第3回 宿題: 学習ログの保存先（固定。モデルにパスを決めさせない）
LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回（1始まりの整数）の用語から確認問題を1問作り、問題と答えを返す。

    「第2回の確認問題を出して」「復習のクイズを出して」のように、
    確認問題・クイズ・理解度チェックを求められたときに使う。
    答えは利用者が回答するまで見せないこと。
    """
    candidates = [e for e in load_glossary() if e["lecture"] == lecture]
    if not candidates:
        available = sorted({e["lecture"] for e in load_glossary()})
        return (
            f"第{lecture}回の用語は用語集にありません。"
            f"問題を作れる回: {', '.join(f'第{n}回' for n in available)}"
        )
    entry = random.choice(candidates)
    return json.dumps(
        {
            "question": f"「{entry['term']}」の意味は？",
            "answer": entry["definition"],
            "lecture": entry["lecture"],
        },
        ensure_ascii=False,
    )


def save_learning_log(title: str, content: str) -> str:
    """学習の記録を Markdown ファイルとして learning_log/ に1件保存する。

    「今日学んだことを記録して」「この内容をメモしておいて」のように、
    学習の記録・メモ・ログを残すよう頼まれたときだけ使う。
    title には記録の短い見出し（例: "LoRAとQLoRAの違い"）を入れる。
    content には会話から学んだ要点を、Markdown の箇条書きでまとめて入れる。
    保存は利用者の承認を得てから実行される。
    """
    title = title.strip()
    content = content.strip()
    if not title or not content:
        return "title と content の両方を指定してください。保存はしていません。"

    now = datetime.now()
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{now:%Y%m%d-%H%M%S}.md"
    text = f"# {title}\n\n- 保存日時: {now:%Y-%m-%d %H:%M:%S}\n\n{content}\n"
    try:
        # "x" は新規作成専用。同じ名前のファイルがあれば上書きせずにエラーにする
        with open(path, "x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError:
        return f"{path.name} はすでにあるため、上書きせずに中止しました。少し待ってから再度保存してください。"
    return f"学習ログを保存しました: learning_log/{path.name}"
