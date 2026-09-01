"""Minimal DeepSeek client for structured, human-reviewed factor proposals."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any

from .formula import validate_formula


@dataclass
class FactorProposal:
    name: str
    formula: str
    intuition: str
    required_columns: list[str]
    validation_plan: list[str]
    risk_notes: list[str]
    model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("DeepSeek response must be a JSON object")
    return value


class DeepSeekClient:
    """A no-dependency DeepSeek Chat Completions client.

    The client only proposes metadata and an allow-listed formula. It has no
    repository or market-data access and never executes model-generated code.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 45.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def propose_factor(
        self, question: str, available_columns: list[str]
    ) -> FactorProposal:
        if not question.strip():
            raise ValueError("question must not be empty")
        schema = {
            "name": "short_reversal_5d",
            "formula": "rank(reversal(close,5))",
            "intuition": "one sentence",
            "required_columns": ["close"],
            "validation_plan": [
                "compute daily rank IC",
                "run a cost-aware long-short backtest",
            ],
            "risk_notes": ["turnover", "liquidity"],
        }
        system = (
            "You are a quantitative equity researcher. Propose one testable factor. "
            "Return JSON only, matching the example keys. Use only the safe formula grammar: "
            "column, -column, return(column,N), momentum(column,N), reversal(column,N), "
            "volatility(column,N), rank(expr), or zscore(expr). Never use future data, code, or eval."
        )
        user = json.dumps(
            {
                "question": question,
                "available_columns": available_columns,
                "example_json": schema,
            },
            ensure_ascii=False,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 800,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
            proposal_data = _parse_json_content(content)
            formula = validate_formula(str(proposal_data["formula"]))
            required = [
                str(value) for value in proposal_data.get("required_columns", [])
            ]
            unknown = set(required) - set(available_columns)
            if unknown:
                raise ValueError(
                    f"proposal uses unavailable columns: {', '.join(sorted(unknown))}"
                )
            return FactorProposal(
                name=str(proposal_data["name"])[:80],
                formula=formula,
                intuition=str(proposal_data.get("intuition", ""))[:500],
                required_columns=required,
                validation_plan=[
                    str(value) for value in proposal_data.get("validation_plan", [])
                ][:10],
                risk_notes=[
                    str(value) for value in proposal_data.get("risk_notes", [])
                ][:10],
                model=self.model,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid factor proposal: {exc}") from exc


def client_from_environment() -> DeepSeekClient | None:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    return DeepSeekClient(
        key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
