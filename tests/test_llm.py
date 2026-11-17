import json
import sys
from types import SimpleNamespace

from bosch_xai import llm

SLICE = {
    "slice": 2,
    "parts": 135,
    "failures": 63,
    "failure_rate": 0.4667,
    "flagged": 135,
    "precision": None,
    "top_features": [{"feature": "L3_S50_F300", "mean_shap": 1.982}],
    "top_stations": [{"station": "L3_S50", "mean_shap": -2.184}],
}


def test_faithful_text_passes():
    text = (
        "Slice 2 holds 135 parts, 63 of which failed (46.7%, or 0.47 as a rate). "
        "L3_S50_F300 raises the risk by 1.98 log-odds, and station L3_S50 lowers it by 2.184."
    )
    assert llm.check_faithfulness(text, SLICE) == []


def test_invented_names_and_numbers_are_flagged():
    text = "Station L0_S12 shows 140 parts, and feature L3_S50_F301 has a 12% failure rate."
    assert llm.check_faithfulness(text, SLICE) == [
        "unknown name L0_S12",
        "unknown name L3_S50_F301",
        "number 140 not in slice",
        "number 12% not in slice",
    ]


def test_partial_station_names_do_not_slip_through():
    assert llm.check_faithfulness("Check S77 first.", SLICE) == ["number 77 not in slice"]


def test_explain_slices_sends_only_the_slice(monkeypatch):
    sent = []

    class FakeMessages:
        def create(self, **kwargs):
            sent.append(kwargs)
            block = SimpleNamespace(type="text", text="Slice 2 has 135 parts and 999 failures.")
            return SimpleNamespace(content=[block], stop_reason="end_turn")

    # A stand-in module, so the test also runs without the optional 'llm' extra installed.
    fake = SimpleNamespace(
        Anthropic=lambda: SimpleNamespace(messages=FakeMessages()), AuthenticationError=OSError
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    slices = [dict(SLICE)]

    llm.explain_slices(slices)

    assert sent[0]["model"] == "claude-haiku-5-5"
    assert sent[0]["messages"][0]["content"] == json.dumps(SLICE, indent=2)
    assert slices[0]["faithfulness_violations"] == ["number 999 not in slice"]
