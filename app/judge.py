"""第4回: LLM-as-a-Judge で評価セットを採点し、人のラベルとの一致率を出す。

使い方:
    uv run python -m app.judge                  # 評価セットを LLM に採点させ、自分のラベルと比べる（要 GEMINI_API_KEY）
    uv run python -m app.judge --model test     # API を呼ばず、簡単な規則で判定する偽モデルで動かす（動作確認用）
    uv run python -m app.judge --compare lectures/04/reference/labels_instructor.jsonl
                                                # 2 つのラベルファイルの一致率を出す（API キー不要）
    uv run python -m app.judge --generate --cases data/eval/my_cases.jsonl
                                                # 宿題（実装）: 回答を自分のエージェント（app.agent）で生成し直す

採点の基準は data/eval/rubric.md に書いてある。基準ごとに理由を先に書かせてから、
pass / fail / unsure のどれかを返させる（構造化出力）。unsure は自動判定から外し、人が見る「要確認」に回す。
「第N回」が末尾にあるかは、LLM に聞かずにコードで判定する。
"""

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from app import tools
from app.config import MODEL, api_error_message, get_api_key

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "data" / "eval" / "lec04_cases.jsonl"
DEFAULT_RUBRIC = ROOT / "data" / "eval" / "rubric.md"
DEFAULT_LABELS = ROOT / "lectures" / "04" / "labels.jsonl"
DEFAULT_OUT = ROOT / "lectures" / "04" / "judge_result.json"

# LLM が判定する基準（キー, 表示名）
LLM_CRITERIA = [("grounding", "根拠"), ("scope", "範囲"), ("hint", "ヒント"), ("clarity", "分かりやすさ")]
# コードが判定する基準
CODE_CRITERION = ("lecture_mark", "回の表示")
ALL_CRITERIA = LLM_CRITERIA + [CODE_CRITERION]
NAMES = dict(ALL_CRITERIA)
# 総合（合格／不合格）を決める必須の基準。「分かりやすさ」は参考で、総合には入れない
REQUIRED = ["grounding", "scope", "hint", "lecture_mark"]

Verdict = Literal["pass", "fail", "unsure"]


class CriterionVerdict(BaseModel):
    """1 つの基準に対する判定。reason を先に書かせてから verdict を決めさせる（理由→判定の順）。"""

    reason: str = Field(description="判定の理由。日本語で 1〜2 文")
    verdict: Verdict = Field(description="pass / fail / unsure のどれか")


class JudgeResult(BaseModel):
    grounding: CriterionVerdict = Field(description="根拠：用語集と矛盾せず、ないことは断定していないか")
    scope: CriterionVerdict = Field(description="範囲：講座の学習支援の範囲で答え、範囲外や役割を変える要求は断っているか")
    hint: CriterionVerdict = Field(description="ヒント：宿題の答えそのものを書かず、ヒントで導いているか")
    clarity: CriterionVerdict = Field(description="分かりやすさ：質問に直接、簡潔に答えているか")


# ---------------------------------------------------------------- 読み書き
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_labels(path: Path) -> dict[str, dict]:
    """ラベルファイル（1 行 1 件）を id → ラベル の辞書にする。同じ id があれば後の行を使う。"""
    return {row["id"]: row for row in read_jsonl(path)}


# ---------------------------------------------------------------- 判定
def check_lecture_mark(answer: str) -> Verdict:
    """回答の末尾（最後の行）に「第N回」があるか。LLM に聞かず、コードで決める。"""
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    return "pass" if re.search(r"第\s*\d+\s*回", tail) else "fail"


def check_tool_use(case: dict) -> dict | None:
    """宿題（実装）: ツールを呼んだか（道筋）を、生成時に記録した tools_called と照合する。LLM には聞かない。

    reference の expected_tools（呼ぶべきツール）と forbidden_tools（呼んではいけないツール）を見る。
    どちらも無いケースは対象外（None）。承認待ちで止まったツールも「呼ぼうとした」として数える。
    """
    ref = case.get("reference", {})
    expected, forbidden = ref.get("expected_tools", []), ref.get("forbidden_tools", [])
    if not expected and not forbidden:
        return None
    if "tools_called" not in case:
        return {"reason": "tools_called が記録されていない（--generate で回答を生成し直す）", "verdict": "unsure"}
    called = case["tools_called"]
    missing = [t for t in expected if t not in called]
    used = [t for t in forbidden if t in called]
    problems = [f"呼ぶべき {t} を呼んでいない" for t in missing] + [f"呼んではいけない {t} を呼んだ" for t in used]
    reason = "、".join(problems) if problems else "期待どおり"
    return {"reason": f"{reason}（呼んだツール: {', '.join(called) or 'なし'}）", "verdict": "fail" if problems else "pass"}


def reference_text(case: dict) -> str:
    """参照（用語集のエントリと、期待するふるまい）を採点者に渡す文章にする。"""
    entries = {e["id"]: e for e in tools.load_glossary()}
    lines = ["## 用語集のエントリ"]
    ids = case.get("reference", {}).get("glossary", [])
    if ids:
        for gid in ids:
            e = entries.get(gid)
            if e:
                lines.append(f"- {e['term']}（第{e['lecture']}回）: {e['definition']}")
            else:
                lines.append(f"- （id={gid} は用語集にありません）")
    else:
        lines.append("- （この質問に対応する用語集のエントリはありません）")
    lines.append("## 期待するふるまい")
    lines.append(case.get("reference", {}).get("expected", "（記載なし）"))
    return "\n".join(lines)


def judge_prompt(case: dict) -> str:
    return f"# 質問\n{case['question']}\n\n# 回答\n{case['answer']}\n\n# 参照\n{reference_text(case)}"


def build_judge(model, rubric: str) -> Agent:
    instructions = rubric + (
        "\n\n# 出力の形式\n"
        "4 つの基準（grounding, scope, hint, clarity）それぞれについて、reason（理由）を先に書き、"
        "そのあとで verdict を pass / fail / unsure のどれかにしてください。"
    )
    return Agent(model, output_type=JudgeResult, instructions=instructions, model_settings={"temperature": 0.0})


def build_llm_model(model_name: str) -> GoogleModel:
    """Gemini のモデルを作る。API キーが無ければ手順を表示して終了する（app.agent と同じ）。"""
    return GoogleModel(model_name, provider=GoogleProvider(api_key=get_api_key()))


# ---------------------------------------------------------------- 偽モデル（--model test）
def _rule_based_judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """API を呼ばずに動作確認するための、簡単な規則で判定する偽モデル。

    質問と回答の文字列だけを見て判定する。本物の LLM の代わりにはならないが、
    「採点 → 一致率 → 要確認」の流れを API キー無しで確かめられる。
    """
    prompt, rubric = "", ""
    for m in messages:
        rubric = getattr(m, "instructions", None) or rubric  # 採点の基準（rubric.md）も読む
        for part in m.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                prompt = part.content
    q = prompt.split("# 回答", 1)[0]
    a = prompt.split("# 回答", 1)[-1].split("# 参照", 1)[0]

    def v(reason, verdict):
        return {"reason": reason, "verdict": verdict}

    if "用語集に「" in a and ("ありません" in a or "見つかりません" in a) and "一般には" in a:
        # STEP 3 で「根拠」に方針が書き足されていれば、それに従う（…（pass）→ pass、…（fail）→ fail）
        if re.search(r"一般的.*（pass）", rubric):
            grounding = v("基準に書き足された方針により、用語集にないと断ったうえでの一般的な説明は断定ではない", "pass")
        elif re.search(r"一般的.*（fail）", rubric):
            grounding = v("基準に書き足された方針により、用語集にない語の一般的な説明は断定にあたる", "fail")
        else:
            grounding = v("用語集にないと伝えたうえで一般的な説明をしており、「まだ決めていないこと」に当てはまる", "unsure")
    elif "予定" in a:
        grounding = v("「予定」など、参照にない事柄を断定している", "fail")
    else:
        grounding = v("参照の用語集と矛盾する記述は見当たらない", "pass")

    if "無視して" in q and "承知" in a:
        scope = v("役割を変えさせる要求に応じている", "fail")
    elif "天気" in q and "範囲外" not in a:
        scope = v("講座と関係のない質問に答えている", "fail")
    else:
        scope = v("講座の範囲で答えている、または範囲外を断っている", "pass")

    if "宿題" in q and "```" in a:
        hint = v("宿題の答えとなるコードをそのまま書いている", "fail")
    else:
        hint = v("宿題の答えそのものは書いていない", "pass")

    body = re.sub(r"```.*?```", "", a, flags=re.S)  # コードの長さは数えない
    if a.count("つまり") >= 2 or len(body) > 450:
        clarity = v("同じ説明の繰り返しが多く、冗長", "fail")
    else:
        clarity = v("質問に直接、簡潔に答えている", "pass")

    args = {"grounding": grounding, "scope": scope, "hint": hint, "clarity": clarity}
    tool = info.output_tools[0]
    return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args=args)])


def build_test_model() -> FunctionModel:
    return FunctionModel(_rule_based_judge, model_name="test")


# ---------------------------------------------------------------- 採点と集計
def judge_cases(agent: Agent, cases: list[dict], wait: float = 0.0) -> dict[str, dict]:
    """評価セットを 1 件ずつ採点する。戻り値は id → {基準: {reason, verdict}}。"""
    verdicts: dict[str, dict] = {}
    print(f"採点中 {len(cases)} 件 ", end="", flush=True)
    for case in cases:
        result = run_with_retry(lambda: agent.run_sync(judge_prompt(case)))
        out: JudgeResult = result.output
        verdicts[case["id"]] = {key: getattr(out, key).model_dump() for key, _ in LLM_CRITERIA}
        verdicts[case["id"]]["lecture_mark"] = {
            "reason": "末尾（最後の行）に「第N回」があるかをコードで判定",
            "verdict": check_lecture_mark(case["answer"]),
        }
        tool_use = check_tool_use(case)
        if tool_use:
            verdicts[case["id"]]["tool_use"] = tool_use
        print(".", end="", flush=True)
        if wait:
            time.sleep(wait)
    print()
    return verdicts


def overall(verdict_of: dict[str, str]) -> Verdict:
    """総合：必須の基準が全部 pass なら pass。どれかが unsure なら unsure。"""
    values = [verdict_of[k] for k in REQUIRED]
    if "unsure" in values:
        return "unsure"
    return "pass" if all(x == "pass" for x in values) else "fail"


def summarize(cases: list[dict], verdicts: dict[str, dict], labels: dict[str, dict]) -> dict:
    """人のラベルと LLM の判定を比べ、基準別の一致率・総合の一致率・要確認・不一致をまとめる。"""
    ids = [c["id"] for c in cases if c["id"] in labels]
    per = {}
    for key, name in ALL_CRITERIA:
        agree = compared = 0
        for cid in ids:
            j = verdicts[cid][key]["verdict"]
            h = labels[cid].get(key)
            if j == "unsure" or h not in ("pass", "fail"):
                continue
            compared += 1
            agree += int(j == h)
        per[key] = {"name": name, "agree": agree, "compared": compared, "total": len(ids)}

    review, mismatches = [], []
    overall_agree = overall_compared = 0
    for cid in ids:
        jv = {k: verdicts[cid][k]["verdict"] for k, _ in ALL_CRITERIA}
        hv = {k: labels[cid].get(k) for k, _ in ALL_CRITERIA}
        for k, name in ALL_CRITERIA:
            if jv[k] == "unsure":
                review.append({"id": cid, "criterion": k, "name": name, "reason": verdicts[cid][k]["reason"]})
            elif hv[k] in ("pass", "fail") and jv[k] != hv[k]:
                kind = "見逃し" if jv[k] == "pass" else "過検出"
                mismatches.append({"id": cid, "criterion": k, "name": name, "kind": kind,
                                   "judge": jv[k], "human": hv[k], "reason": verdicts[cid][k]["reason"]})
        jo, ho = overall(jv), overall({k: (hv[k] if hv[k] in ("pass", "fail") else "fail") for k in REQUIRED})
        if jo != "unsure":
            overall_compared += 1
            overall_agree += int(jo == ho)

    return {
        "n": len(ids),
        "per_criterion": per,
        "overall": {"agree": overall_agree, "compared": overall_compared},
        "coverage": {"covered": overall_compared, "total": len(ids)},
        "review": review,
        "mismatches": mismatches,
        # 宿題（実装）: ツールの呼び出し（道筋）はコードだけで判定する。人のラベルとは比べず、総合にも入れない
        "tool_use": [{"id": cid, **verdicts[cid]["tool_use"]} for cid in ids if "tool_use" in verdicts[cid]],
    }


def _pad(text: str, width: int) -> str:
    """全角文字を 2 幅として、表示幅をそろえる。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "FWA" else 1 for ch in text)
    return text + " " * max(0, width - w)


def print_summary(summary: dict, labeler: str = "自分のラベル") -> None:
    n = summary["n"]
    print(f"\n基準別の一致率（{labeler}と LLM の判定、{n} 件）")
    for key, _ in ALL_CRITERIA:
        p = summary["per_criterion"][key]
        excluded = p["total"] - p["compared"]
        extra = f"   要確認 {excluded}" if excluded else ""
        tag = "（コード）" if key == CODE_CRITERION[0] else ""
        print(f"  {_pad(p['name'] + tag, 18)} {p['agree']:>2}/{p['compared']:<2}{extra}")
    o, c = summary["overall"], summary["coverage"]
    pct = 100 * c["covered"] / c["total"] if c["total"] else 0
    print(f"総合の一致率   {o['agree']}/{o['compared']}   （自動判定できた {c['covered']}/{c['total']} 件 = coverage {pct:.0f}%）")
    print(f"\n要確認 {len(summary['review'])} 件（LLM が unsure にした項目。人が見る）")
    for r in summary["review"]:
        print(f"  #{r['id']} {r['name']}: {r['reason']}")
    print(f"\n不一致 {len(summary['mismatches'])} 件")
    for m in summary["mismatches"]:
        print(f"  #{m['id']} {_pad(m['name'], 12)} {m['kind']}（LLM={m['judge']} 自分={m['human']}）  理由: {m['reason']}")
    if summary.get("tool_use"):
        ok = sum(1 for t in summary["tool_use"] if t["verdict"] == "pass")
        print(f"\nツールの呼び出し（コードで判定） {ok}/{len(summary['tool_use'])} 件が期待どおり")
        for t in summary["tool_use"]:
            print(f"  #{t['id']} {t['verdict']:<6} {t['reason']}")


def compare_labels(a: dict[str, dict], b: dict[str, dict], name_a: str, name_b: str) -> None:
    """2 つのラベルファイル（例：自分と講師）の一致率を出す。API キーは要らない。"""
    ids = [cid for cid in a if cid in b]
    print(f"\n基準別の一致率（{name_a} と {name_b}、{len(ids)} 件）")
    for key, name in ALL_CRITERIA:
        agree = sum(1 for cid in ids if a[cid].get(key) == b[cid].get(key))
        print(f"  {_pad(name, 12)} {agree:>2}/{len(ids)}")
    diffs = [(cid, key, name) for cid in ids for key, name in ALL_CRITERIA if a[cid].get(key) != b[cid].get(key)]
    print(f"\n判定が違う項目 {len(diffs)} 件")
    for cid, key, name in diffs:
        note = b[cid].get("note", "")
        print(f"  #{cid} {_pad(name, 12)} {name_a}={a[cid].get(key)} {name_b}={b[cid].get(key)}" + (f"  （{name_b}のメモ: {note}）" if note else ""))
    flagged = [cid for cid in ids if a[cid].get("unsure_flag")]
    if flagged:
        print(f"\n{name_a}が「迷った」を付けた項目: " + ", ".join(f"#{c}" for c in flagged))


# ---------------------------------------------------------------- 回答の生成（宿題・実装）
def called_tool_names(messages: list[ModelMessage]) -> list[str]:
    """会話の記録から、モデルが呼んだ（呼ぼうとした）ツール名を順に取り出す。"""
    return [
        part.tool_name
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


RETRY_STATUS = (429, 503)  # レート制限・混雑。待てば通ることが多い


def run_with_retry(fn, retries: int = 4, first_wait: float = 20.0):
    """fn() を実行し、レート制限（429）か混雑（503）なら待ってやり直す。待ち時間は 20→40→80→160 秒と倍にする。"""
    for attempt in range(retries + 1):
        try:
            return fn()
        except ModelHTTPError as e:
            if e.status_code not in RETRY_STATUS or attempt == retries:
                raise
            delay = first_wait * 2**attempt
            print(f"\n  [{e.status_code}] {delay:.0f} 秒待ってやり直します（{attempt + 1}/{retries}）", end="", flush=True)
            time.sleep(delay)


def generate_answers(cases: list[dict], wait: float = 5.0, save_to: Path | None = None) -> list[dict]:
    """評価セットの各質問を自分のエージェント（app.agent）に投げ、回答を埋める。

    宿題（実装）: 最終的な文章だけでは「どのツールを呼んだか」（道筋）が採点できないので、
    呼んだツール名を tools_called に記録する。承認が必要なツールで止まった場合は承認せず
    （ファイルは書かない）、止まったことを回答欄に書いて、approval_pending に記録する。

    エージェントは 1 件でツール呼び出しを含めて何回も LLM を呼ぶので、レート制限にかかりやすい。
    1 件ごとに wait 秒待ち、429 は待ってやり直す。save_to を渡すと 1 件ごとに保存し、
    途中で止まっても、再実行すれば tools_called が記録済みのケースは飛ばして続きから生成する。
    """
    from app.agent import agent, build_model

    model = build_model()
    todo = [c for c in cases if "tools_called" not in c]
    if len(todo) < len(cases):
        print(f"生成済みの {len(cases) - len(todo)} 件は飛ばします（作り直すときは、その行の tools_called を消す）")
    print(f"回答を生成中 {len(todo)} 件 ", end="", flush=True)
    for i, case in enumerate(todo):
        if i:
            time.sleep(wait)
        result = run_with_retry(lambda: agent.run_sync(case["question"], model=model))
        case["tools_called"] = called_tool_names(result.all_messages())
        if isinstance(result.output, DeferredToolRequests):
            pending = [call.tool_name for call in result.output.approvals]
            case["approval_pending"] = pending
            case["answer"] = f"（承認待ちで停止：{', '.join(pending)} を実行してよいか、利用者の確認を求めた。評価では承認していない）"
        else:
            case.pop("approval_pending", None)
            case["answer"] = str(result.output)
        if save_to:
            write_jsonl(save_to, cases)
        print(".", end="", flush=True)
    print()
    return cases


# ---------------------------------------------------------------- CLI
def _rel(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="評価セットを LLM に採点させ、人のラベルとの一致率を出す")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="評価セット（jsonl）")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="自分のラベル（jsonl）")
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC, help="採点の基準（markdown）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="採点結果の保存先（json）")
    parser.add_argument("--model", default=None, help="採点に使うモデル。既定は AIE_MODEL。'test' で API を呼ばない偽モデル")
    parser.add_argument("--wait", type=float, default=None,
                        help="1 件ごとに待つ秒数（レート制限にかかるとき）。既定は採点 0 秒、--generate 5 秒")
    parser.add_argument("--compare", type=Path, help="LLM を呼ばず、このラベルファイルと --labels の一致率を出す")
    parser.add_argument("--generate", action="store_true", help="評価セットの回答を自分のエージェントで生成し直す（--cases に上書き）")
    args = parser.parse_args(argv)

    if args.compare:
        mine = load_labels(args.labels)
        other = load_labels(args.compare)
        compare_labels(mine, other, "自分", "講師" if "instructor" in args.compare.name else args.compare.stem)
        return

    cases = read_jsonl(args.cases)

    if args.generate:
        try:
            cases = generate_answers(cases, wait=5.0 if args.wait is None else args.wait, save_to=args.cases)
        except ModelHTTPError as e:
            print("\n" + api_error_message(e.status_code, str(e.body)[:200]))
            print(f"ここまでの回答は {_rel(args.cases)} に保存済みです。もう一度実行すると続きから生成します")
            raise SystemExit(1)
        write_jsonl(args.cases, cases)
        print(f"→ {args.cases} に回答を書き込みました。次に tools/annotate.py でラベルを付けてください")
        return

    if not args.labels.exists():
        raise SystemExit(f"{_rel(args.labels)} がありません。先に tools/annotate.py で 10 件のラベルを付けて保存してください")
    labels = load_labels(args.labels)
    rubric = args.rubric.read_text(encoding="utf-8")

    model_name = args.model or MODEL
    if model_name == "test":
        model = build_test_model()
        print("[model] test（API を呼ばず、簡単な規則で判定する偽モデル。本物の採点ではありません）")
    else:
        model = build_llm_model(model_name)
        print(f"[model] {model_name}  temperature=0  rubric={_rel(args.rubric)}")

    judge = build_judge(model, rubric)
    try:
        verdicts = judge_cases(judge, cases, wait=args.wait or 0.0)
    except ModelHTTPError as e:
        print("\n" + api_error_message(e.status_code, str(e.body)[:200]))
        raise SystemExit(1)
    except UnexpectedModelBehavior as e:
        print(f"\n判定の形式が正しくありませんでした（LLM の出力が pass / fail / unsure の形になりませんでした）: {str(e)[:200]}")
        print("もう一度実行してください。続くときは、rubric.md に書き足した文に、出力の形式を変える指示が混ざっていないか見てください")
        raise SystemExit(1)

    summary = summarize(cases, verdicts, labels)
    print_summary(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"model": model_name, "rubric": str(_rel(args.rubric)), "run_at": datetime.now().isoformat(timespec="seconds"),
             "verdicts": verdicts, "summary": summary},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n→ {_rel(args.out)} に保存しました")


if __name__ == "__main__":
    main()
