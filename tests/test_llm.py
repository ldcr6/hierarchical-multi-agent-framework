from hmas.config import ModelConfig
from hmas.llm import LLMClient, LLMResponse, Usage


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    cfg = ModelConfig(name="fake", base_url="http://localhost:1", model="fake", api_key_env="FAKE_KEY")
    c = LLMClient(cfg, cache_path=tmp_path / "c.sqlite")
    calls = []

    def fake_call(messages, temperature, max_tokens):
        calls.append(messages)
        return LLMResponse("```python\ndef f():\n    return 1\n```", 10, 5, 0.1)

    monkeypatch.setattr(c, "_call", fake_call)
    return c, calls


def test_cache_and_usage(tmp_path, monkeypatch):
    c, calls = make_client(tmp_path, monkeypatch)
    msgs = [{"role": "user", "content": "hi"}]
    u = Usage()
    r1 = c.chat(msgs, usage=u)
    r2 = c.chat(msgs, usage=u)
    assert len(calls) == 1 and not r1.cached and r2.cached
    assert u.calls == 2 and u.prompt_tokens == 20 and u.cached_calls == 1
    c.chat(msgs, sample_index=1)  # a different sample is a new call
    assert len(calls) == 2


def test_cost():
    cfg = ModelConfig(name="m", base_url="", model="m", api_key_env="X", price_in=2.0, price_out=8.0)
    assert cfg.cost(1_000_000, 500_000) == 6.0
