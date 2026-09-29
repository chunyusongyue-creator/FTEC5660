#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create independent extraction, review, and judge chains."""
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=2,
    )

    # Pass 1: extract all values needed for both queries.
    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert at reading Hong Kong supermarket receipts.

Inspect the ORIGINAL receipt image and extract:

1. final_payment
   - The actual amount paid AFTER the receipt's ROUNDING adjustment.
   - Usually the final tender/payment amount such as CASH, OCTOPUS, VISA,
     MASTER, EPS, ALIPAY, WECHAT PAY, etc.
   - Do not use SUBTOTAL when ROUNDING changes the actual payment.

2. subtotal
   - The exact printed SUBTOTAL.

3. discounts
   - Every explicit discount, promotion, coupon, member/app discount,
     percentage discount, packaging-damage discount, markdown, or other
     qualifying reduction that must be added back to SUBTOTAL.
   - Return each discount as a POSITIVE amount.
   - ROUNDING is NOT a discount.
   - Preserve separate discount lines.

Return STRICT JSON only:

{{
  "final_payment": "102.30",
  "subtotal": "102.31",
  "discounts": ["5.39"]
}}

Use exactly two decimal places. Do not include HK$, $, commas, Markdown,
or explanatory text. Do not invent values.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Pass 2: blind independent review of both Query 1 and Query 2 fields.
    review_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a BLIND independent supermarket receipt reviewer.

You are NOT given another model's answer. Inspect the ORIGINAL receipt image
from scratch and independently determine:

1. final_payment
   - The actual amount paid AFTER ROUNDING.

2. subtotal
   - The exact printed SUBTOTAL.

3. discounts
   - Every qualifying discount/promotion/coupon amount that must be added
     back to SUBTOTAL.
   - Return all discounts as POSITIVE values.
   - Do NOT include ROUNDING.

4. without_discount
   - subtotal + sum(discounts)

Return STRICT JSON only:

{{
  "final_payment": "102.30",
  "subtotal": "102.31",
  "discounts": ["5.39"],
  "without_discount": "107.70"
}}

Use exactly two decimal places. Do not include HK$, $, commas, Markdown,
or explanatory text. Do not invent values.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Pass 3: resolve disagreements or recover if one earlier pass fails.
    judge_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the final receipt judge.

Inspect the ORIGINAL receipt image carefully. Candidate values may be supplied,
but they may be wrong. Determine the final correct values yourself:

1. final_payment: actual amount paid AFTER ROUNDING
2. subtotal: exact printed SUBTOTAL
3. discounts: all qualifying discount/promotion/coupon amounts as POSITIVE values
4. without_discount: subtotal + sum(discounts)

ROUNDING must never be included in discounts.

Return STRICT JSON only:

{{
  "final_payment": "102.30",
  "subtotal": "102.31",
  "discounts": ["5.39"],
  "without_discount": "107.70"
}}

Use exactly two decimal places. Do not include HK$, $, commas, Markdown,
or explanatory text. Do not invent values.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return {
        "extractor": extraction_prompt | model,
        "reviewer": review_prompt | model,
        "judge": judge_prompt | model,
    }


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run robust extraction/review for both queries and aggregate results."""
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage

    extractor = chain["extractor"]
    reviewer = chain["reviewer"]
    judge = chain["judge"]

    def to_decimal(value: Any) -> Decimal:
        """Convert a model monetary value to Decimal."""
        if value is None:
            raise ValueError("Missing monetary value")
        text = str(value).strip().replace("HK$", "").replace("$", "").replace(",", "")
        return Decimal(text).quantize(Decimal("0.01"))

    def parse_json_object(value: Any) -> dict[str, Any]:
        """Parse JSON while tolerating harmless Markdown fences."""
        text = response_text(value).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data

        raise ValueError(f"Could not parse model JSON: {text}")

    def normalize_discounts(values: Any) -> list[Decimal]:
        """Normalize discounts for comparison and arithmetic."""
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]

        discounts = []
        for value in values:
            amount = abs(to_decimal(value))
            if amount != Decimal("0.00"):
                discounts.append(amount)
        return sorted(discounts)

    def calculate_without_discount(
        subtotal: Decimal,
        discounts: list[Decimal],
    ) -> Decimal:
        """Calculate Query 2 for one receipt."""
        return (
            subtotal + sum(discounts, Decimal("0.00"))
        ).quantize(Decimal("0.01"))

    def build_image_message(image: Path, text: str) -> HumanMessage:
        """Build one multimodal message."""
        return HumanMessage(
            content=[
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image)},
                },
            ]
        )

    def parse_extractor_result(value: Any) -> dict[str, Any]:
        """Validate the extractor result."""
        data = parse_json_object(value)
        subtotal = to_decimal(data["subtotal"])
        discounts = normalize_discounts(data.get("discounts", []))
        return {
            "final_payment": to_decimal(data["final_payment"]),
            "subtotal": subtotal,
            "discounts": discounts,
            "without_discount": calculate_without_discount(subtotal, discounts),
        }

    def parse_full_result(value: Any) -> dict[str, Any]:
        """Validate reviewer/judge output for both queries."""
        data = parse_json_object(value)
        subtotal = to_decimal(data["subtotal"])
        discounts = normalize_discounts(data.get("discounts", []))
        calculated = calculate_without_discount(subtotal, discounts)

        if "without_discount" in data:
            if to_decimal(data["without_discount"]) != calculated:
                raise ValueError("Inconsistent without_discount")

        return {
            "final_payment": to_decimal(data["final_payment"]),
            "subtotal": subtotal,
            "discounts": discounts,
            "without_discount": calculated,
        }

    def results_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """Require agreement on both Query 1 and Query 2 values."""
        return (
            a["final_payment"] == b["final_payment"]
            and a["subtotal"] == b["subtotal"]
            and a["discounts"] == b["discounts"]
            and a["without_discount"] == b["without_discount"]
        )

    extract_inputs = []
    review_inputs = []

    for image in images:
        extract_inputs.append(
            {
                "messages": [
                    build_image_message(
                        image,
                        "Extract final_payment, subtotal, and all qualifying discounts.",
                    )
                ]
            }
        )

        review_inputs.append(
            {
                "messages": [
                    build_image_message(
                        image,
                        "Blindly review this receipt from scratch for both queries.",
                    )
                ]
            }
        )

    # Keep batch execution from failing the entire folder because of one request.
    extract_results = extractor.batch(extract_inputs, return_exceptions=True)
    review_results = reviewer.batch(review_inputs, return_exceptions=True)

    final_receipts = []

    for image, extract_result, review_result in zip(
        images,
        extract_results,
        review_results,
    ):
        extracted = None
        reviewed = None

        if not isinstance(extract_result, Exception):
            try:
                extracted = parse_extractor_result(extract_result)
            except (
                ValueError,
                InvalidOperation,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ):
                extracted = None

        if not isinstance(review_result, Exception):
            try:
                reviewed = parse_full_result(review_result)
            except (
                ValueError,
                InvalidOperation,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ):
                reviewed = None

        # If both independent passes agree, use the reviewed result.
        if extracted is not None and reviewed is not None and results_match(
            extracted,
            reviewed,
        ):
            final_receipts.append(reviewed)
            continue

        # Disagreement or one failed pass: invoke the judge.
        candidate_a = (
            {
                "final_payment": f"{extracted['final_payment']:.2f}",
                "subtotal": f"{extracted['subtotal']:.2f}",
                "discounts": [f"{x:.2f}" for x in extracted["discounts"]],
                "without_discount": f"{extracted['without_discount']:.2f}",
            }
            if extracted is not None
            else "unavailable"
        )

        candidate_b = (
            {
                "final_payment": f"{reviewed['final_payment']:.2f}",
                "subtotal": f"{reviewed['subtotal']:.2f}",
                "discounts": [f"{x:.2f}" for x in reviewed["discounts"]],
                "without_discount": f"{reviewed['without_discount']:.2f}",
            }
            if reviewed is not None
            else "unavailable"
        )

        judge_message = build_image_message(
            image,
            (
                "Resolve this receipt for both homework queries.\n\n"
                f"Candidate A: {json.dumps(candidate_a)}\n"
                f"Candidate B: {json.dumps(candidate_b)}\n\n"
                "Inspect the original receipt yourself and return the final values."
            ),
        )

        judged = None
        try:
            judged = parse_full_result(
                judge.invoke({"messages": [judge_message]})
            )
        except Exception:
            judged = None

        if judged is not None:
            final_receipts.append(judged)
        elif reviewed is not None:
            # Valid reviewer result is the first fallback.
            final_receipts.append(reviewed)
        elif extracted is not None:
            # Valid extractor result is the second fallback.
            final_receipts.append(extracted)
        else:
            # Last-resort numeric fallback keeps results.csv generation alive
            # even if all model calls for one receipt fail.
            final_receipts.append(
                {
                    "final_payment": Decimal("0.00"),
                    "subtotal": Decimal("0.00"),
                    "discounts": [],
                    "without_discount": Decimal("0.00"),
                }
            )

    total_spent = sum(
        (receipt["final_payment"] for receipt in final_receipts),
        Decimal("0.00"),
    ).quantize(Decimal("0.01"))

    total_without_discount = sum(
        (receipt["without_discount"] for receipt in final_receipts),
        Decimal("0.00"),
    ).quantize(Decimal("0.01"))

    return {
        QUERY_1: f"HK${total_spent:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {
        query: Decimal(str(answers[query])).quantize(Decimal("0.01"))
        for query in QUERIES
    }


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow(
                [query, text, correctness_text(text, truth.get(query))]
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FTEC5660 HW1 on receipt images"
    )
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)

    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(
        responses,
        read_ground_truth(args.image_folder)
    )

    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


