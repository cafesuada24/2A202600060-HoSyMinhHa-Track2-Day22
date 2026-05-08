import json
import re
from typing import Any

from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import FailResult, PassResult


# ── 1. Python 3.12 Compiled Regex Patterns ──────────────────────────────────
# Compiling regex patterns once at the module/class level drastically improves
# performance, especially when validations are run repeatedly.
class PIIPatterns:
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

class JSONRepairPatterns:
    FENCE_START = re.compile(r"^```(?:json)?\s*", flags=re.MULTILINE)
    FENCE_END = re.compile(r"\s*```$", flags=re.MULTILINE)
    TRAILING_COMMA = re.compile(r",\s*([}\]])")


# ── 2. PII Detector Validator ───────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detects and redacts Personally Identifiable Information (PII)."""

    def validate(self, value: str, metadata: dict[str, Any] | None = None) -> PassResult | FailResult:
        redacted_text = value
        found_pii: list[tuple[str, str]] = []

        # Iterate through the compiled patterns dynamically
        for pii_type, pattern in vars(PIIPatterns).items():
            if pii_type.startswith("_"):
                continue  # Skip dunder methods
                
            matches = pattern.findall(value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            # Returning FailResult with fix_value triggers the OnFailAction.FIX action
            return FailResult(
                errorMessage=f"Detected PII: {[p[0] for p in found_pii]}",
                fixValue=redacted_text
            )
            
        return PassResult()


# ── 3. JSON Formatter Validator ─────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validates and auto-repairs malformed JSON strings."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()

        # Remove markdown fences using compiled patterns
        text = JSONRepairPatterns.FENCE_START.sub("", text)
        text = JSONRepairPatterns.FENCE_END.sub("", text)
        text = text.strip()

        # Single quotes → double quotes (naive replacement)
        text = text.replace("'", '"')

        # Remove trailing commas before } or ]
        text = JSONRepairPatterns.TRAILING_COMMA.sub(r"\1", text)

        return text

    def validate(self, value: str, metadata: dict[str, Any] | None = None) -> PassResult | FailResult:
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass # Fall through to repair logic

        # Try repair
        repaired_text = self._repair(value)
        try:
            parsed = json.loads(repaired_text)
            print("  🔧 JSON repaired successfully")
            return FailResult(
                errorMessage="Invalid JSON format",
                fixValue=json.dumps(parsed, indent=2)
            )
        except json.JSONDecodeError as e:
            fallback = json.dumps({"error": "JSON unrecoverable", "raw": value})
            return FailResult(
                errorMessage=f"JSON unrecoverable: {e}",
                fixValue=fallback
            )


# ── 4. PII Guard Demo ───────────────────────────────────────────────────────
def demo_pii_guard() -> None:
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 5. JSON Guard Demo ──────────────────────────────────────────────────────
def demo_json_guard() -> None:
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",      '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",   "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",  '{"key": "value",}'),
        ("Truly invalid",   "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        
        # Check Guardrails outcome safely
        if result.validation_passed:
            status = "✅ Pass"
        elif result.validated_output:
            status = "🔧 Fixed"
        else:
            status = "❌ Fail"
            
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text.strip()}")
        print(f"  Output: {result.validated_output}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Step 4 complete!")

if __name__ == "__main__":
    main()
