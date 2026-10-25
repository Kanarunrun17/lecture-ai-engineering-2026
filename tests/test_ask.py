from types import SimpleNamespace

from app.ask import run_interactive


class FakeChat:
    def __init__(self):
        self.messages = []

    def send_message(self, message):
        self.messages.append(message)
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            text=f"回答: {message}",
        )


def test_interactive_session_keeps_sending_follow_up_questions_to_same_chat(monkeypatch, capsys):
    chat = FakeChat()
    questions = iter(["その理由は？", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    run_interactive(chat, "最初の質問")

    assert chat.messages == ["最初の質問", "その理由は？"]
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: その理由は？" in output
