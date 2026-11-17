"""Optional plain-language slice summaries from the Anthropic API, checked for faithfulness."""

import json
import os
import re
import sys

NAME = re.compile(r"L\d+_S\d+(?:_[A-Za-z]+\d*)?")
NUMBER = re.compile(r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?%?")
SYSTEM = (
    "You explain clusters of high-risk parts from a failure prediction model to a production "
    "engineer. You receive one slice as JSON. Write exactly three plain-text sentences. Use "
    "feature and station names exactly as written in the JSON. Use only numbers that appear "
    "in the JSON, copied as written or as a percentage of a rate. mean_shap is the average "
    "contribution to the failure log-odds; positive values raise the predicted risk. Describe "
    "what the data shows and do not invent physical causes."
)


def json_values(obj: object) -> tuple[set[str], list[float]]:
    """All strings and numbers inside a JSON-like object."""
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        names, numbers = set(), []
        for item in obj:
            n, x = json_values(item)
            names |= n
            numbers += x
        return names, numbers
    if isinstance(obj, bool) or obj is None:
        return set(), []
    if isinstance(obj, int | float):
        return set(), [float(obj)]
    return {str(obj)}, []


def check_faithfulness(text: str, slice_: dict) -> list[str]:
    """Every name and number in `text` must come from `slice_`; return the violations.

    A number matches if it equals a JSON value rounded to the decimals written in the text,
    either directly or as a percentage. Signs are ignored because text says "raises"/"lowers".
    """
    names, numbers = json_values(slice_)
    violations = [f"unknown name {n}" for n in NAME.findall(text) if n not in names]
    for token in NUMBER.findall(NAME.sub(" ", text)):
        digits = token.rstrip("%").replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        value = float(digits)
        scale = 100 if token.endswith("%") else 1
        if not any(round(abs(n) * scale, decimals) == value for n in numbers):
            violations.append(f"number {token} not in slice")
    return violations


def explain_slices(slices: list[dict]) -> None:
    """Add `llm_explanation` and `faithfulness_violations` to each slice in place."""
    try:
        import anthropic
    except ImportError:
        sys.exit("The --llm option needs the 'llm' extra: uv sync --extra llm")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("The --llm option needs an API key: export ANTHROPIC_API_KEY=...")
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
    for s in slices:
        payload = dict(s)  # the slice JSON only, nothing else is sent
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                output_config={"effort": "low"},  # a three-sentence summary needs little thinking
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
            )
        except anthropic.AuthenticationError:
            sys.exit("Anthropic authentication failed: set ANTHROPIC_API_KEY.")
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        s["llm_explanation"] = text
        s["faithfulness_violations"] = check_faithfulness(text, payload)
        if response.stop_reason != "end_turn":
            s["faithfulness_violations"].append(f"response stopped: {response.stop_reason}")
        print(f"slice {s['slice']}: {len(s['faithfulness_violations'])} violation(s)")
