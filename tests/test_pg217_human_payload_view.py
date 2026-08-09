from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "research" / "pg217_human_payload_view_v1.md"


def test_pg217_human_view_is_readable_but_redacted() -> None:
    text = VIEW.read_text(encoding="utf-8")
    assert "<RUNTIME_CANARY>" in text
    assert "POST" in text and "GET" in text
    assert "/vul/sqli/sqli_id.php" in text
    assert "/vul/sqli/sqli_str.php" in text
    assert "blind_b" in text and "abstain" in text
    lowered = text.casefold()
    assert "sleep(" not in lowered
    assert "benchmark(" not in lowered
    assert "union select" not in lowered
    assert "原始响应" in text
