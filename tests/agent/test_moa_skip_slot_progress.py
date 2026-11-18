"""Recursion-guard skipped slots must count as terminal fan-out progress.

A slot whose ``provider`` is ``moa`` is skipped on purpose (a MoA preset must not
reference MoA recursively). It still occupies a fan-out slot and always yields a
placeholder, yet before this fix it never advanced ``completed``:

* a mixed fan-out (one ``moa`` slot + one real advisor) reported ``1 of 2``
  forever — the UI showed a permanently incomplete fan-out even though the
  aggregator ran with every slot;
* the turn's activity clock was never stamped for the skipped slot.

Control group: the test asserts both the placeholder text and the final progress
``(2, 2)``. With the fix reverted the placeholder assertion still passes while
the progress assertion fails with ``(1, 2)``, so the assertion is tied to the
fix rather than to general fan-out health.
"""

from types import SimpleNamespace

from agent import moa_loop


def _response():
    message = SimpleNamespace(content="advice", tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def test_skipped_moa_slot_advances_progress(monkeypatch):
    monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: _response())

    seen = []
    slots = [
        {"provider": "moa", "model": "expert-panel"},
        {"provider": "openai-codex", "model": "gpt-5.5"},
    ]

    results = moa_loop._run_references_parallel(
        slots,
        [{"role": "user", "content": "hi"}],
        progress_callback=lambda done, total, label: seen.append((done, total)),
    )

    assert len(results) == 2
    assert "skipped: MoA presets cannot recursively reference MoA" in results[0][1]
    assert seen, "progress_callback never fired"
    assert seen[-1] == (2, 2), f"fan-out progress stuck at {seen[-1]} (expected (2, 2)): {seen}"
