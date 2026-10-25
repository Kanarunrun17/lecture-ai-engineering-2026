"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 回答のあと、続けて質問する

ツールは持たない「素の」呼び出し。--interactive では同じチャットを使い、
会話履歴を保ったまま続けて質問できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    usage = response.usage_metadata
    print(f"[usage] input_tokens={usage.prompt_token_count}, output_tokens={usage.candidates_token_count}")

    return response.text


def ask_in_chat(question: str, chat: Chat) -> str:
    response = chat.send_message(question)

    usage = response.usage_metadata
    print(f"[usage] input_tokens={usage.prompt_token_count}, output_tokens={usage.candidates_token_count}")

    return response.text


def run_interactive(chat: Chat, question: str) -> None:
    while True:
        print(f"[質問] {question}\n")
        print(ask_in_chat(question, chat))
        try:
            question = input("\n続けて質問してください（終了: exit）: ").strip()
        except EOFError:
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--interactive", action="store_true", help="同じ会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            client = genai.Client(api_key=get_api_key())
            chat = client.chats.create(model=MODEL)
            run_interactive(chat, args.question)
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
