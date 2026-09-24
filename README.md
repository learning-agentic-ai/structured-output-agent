# 🛃 The Data Customs Officer: Structured Output Agent Architecture Notes

An engineering blueprint and design manual for technical architects, backend engineers, and AI platform teams. This document formalizes the **Structured Output Agent Pattern**, shifting LLM integration from unconstrained probabilistic text generation to deterministic, schema-enforced transactional boundaries.

---

## 🏛️ System Architecture: The 4-Gate Inspection Lane

Treat all Large Language Model (LLM) completions as uninspected international cargo. Unchecked strings must pass through a strict, 4-gate verification lane before entering internal microservice boundaries.

```
 [ UNSTRUCTURED INGESTION ]                      |         [ TRANSACTION BOUNDARY ]
                                                 |
   Natural Language Prompts                      |   PostgreSQL / MySQL Engines
   Customer Chat Logs / Voice Transcripts        |   Payment Processors (Stripe / Adyen)
             │                                   |   Enterprise ERP & Travel APIs
             ▼                                   |                 ▲
 ═══════════════════  DATA CUSTOMS INSPECTION LANE  ══════════════╡
 │                                                                │
 │  Gate 1: Ingestion Buffer   ──► Untrusted, raw input payload   │
 │  Gate 2: Schema Contract    ──► Pydantic v2 Type Constraints   │
 │  Gate 3: Agent Extraction   ──► Provider Grammar Decoding      │
 │  Gate 4: Runtime Clearance  ──► Object Hydration & Asserts ────┘
 │                                                                │
 ══════════════════════════════════════════════════════════════════

```

* **Gate 1: Ingestion Buffer**
Accepts raw, multi-intent, uncurated natural language inputs without applying brittle, manually maintained regex or string cleaners.
* **Gate 2: Schema Contract (The Rulebook)**
Codifies strict business requirements using **Pydantic v2**: non-nullable fields, closed-set enumerations (`Literal`), explicit primitive and date types, and field validators (`@field_validator`).
* **Gate 3: Agent Interrogation (Constrained Decoding)**
Enforces structural constraints at the LLM provider's token sampling layer using native tool calling. The model's token sampler is bound to the generated JSON schema, eliminating conversational filler and markdown formatting artifacts.
* **Gate 4: Runtime Customs Clearance & Invariant Assertions**
Deserializes and validates payload arguments via `model_validate_json()`. Executes deterministic post-extraction assertions (such as temporal day-of-week validation) before emitting safe objects into core applications.

---

## ⚙️ Core Configuration & Implementation Blocks

These two code blocks set up the provider-agnostic engine and execute the complete 4-gate extraction and clearance lifecycle.

### Code Block 1: Imports & Provider Client Initialization

> **Configuration Note:** Update `GROQ_API_KEY` or `OPENROUTER_API_KEY` with your credentials prior to execution.

```python
import os
import json
from datetime import datetime, date
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from IPython.display import display, HTML, clear_output

# =====================================================================
# ⚙️ PROVIDER TOGGLE: SWITCH BETWEEN GROQ AND OPENROUTER HERE
# =====================================================================
PROVIDER = "groq"  # Supported values: "groq" | "openrouter"

if PROVIDER == "groq":
    from groq import Groq
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "gsk_your_groq_api_key_here")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    MODEL_ID = "llama-3.3-70b-versatile"

elif PROVIDER == "openrouter":
    from openai import OpenAI
    os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "sk-or-v1-your_openrouter_api_key_here")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={"HTTP-Referer": "http://localhost:8888", "X-Title": "Data Customs Officer"}
    )
    MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

print(f"✓ Initialized [{PROVIDER.upper()}] with model: {MODEL_ID}")

```

### Code Block 2: Schema Contract, Agent Extraction & Gate 4 Clearance

```python
# 1. GATE 2: Pydantic Schema Contract
class FlightBooking(BaseModel):
    intent: Literal["flight_booking"] = "flight_booking"
    passenger_name: str
    origin: str
    destination: str
    raw_date_expression: str
    departure_date: date
    seat_preference: Literal["aisle_quiet", "aisle", "window", "middle"]
    budget_limit_usd: int
    meal: Literal["vegan", "standard", "none"]

    @field_validator("departure_date")
    @classmethod
    def verify_not_past(cls, v: date) -> date:
        if v < date.today():
            raise ValueError(f"Departure date {v} cannot be in the past.")
        return v

flight_schema = {
    "name": "submit_flight_booking",
    "description": "Records and validates a clean flight booking payload with exact origin, destination, and calendar dates.",
    "parameters": FlightBooking.model_json_schema()
}

# 2. GATES 1 & 3: Temporal Anchoring & Provider Grammar Extraction
def run_customs_officer(raw_user_text: str) -> str:
    now = datetime.now()
    system_instruction = f"""
You are an expert flight reservation customs extraction officer.
TEMPORAL GROUND TRUTH ANCHOR:
- Today's date is: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}).
- All relative dates (e.g., 'next Friday', 'tomorrow') MUST be calculated starting from this reference date.
- Calculate the exact ISO-8601 date string (YYYY-MM-DD) for 'departure_date'.
Invoke the submit_flight_booking function with your extracted arguments.
"""
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_user_text}
        ],
        tools=[{"type": "function", "function": flight_schema}],
        tool_choice={"type": "function", "function": {"name": "submit_flight_booking"}},
        temperature=0.0
    )
    return response.choices[0].message.tool_calls[0].function.arguments

# 3. GATE 4: Runtime Enforcement, Calendar Verification & Native Deserialization
user_input = "Book me a flight from San Francisco to Tokyo next Friday under 900 bucks, make sure it's an aisle seat and a vegan meal. Name is Jaichand."

try:
    raw_payload = run_customs_officer(user_input)
    validated = FlightBooking.model_validate_json(raw_payload)
    
    # Deterministic Business Gate: Verify weekday alignment
    actual_weekday = validated.departure_date.strftime("%A")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        if day.lower() in validated.raw_date_expression.lower() and actual_weekday != day:
            raise ValueError(f"Calendar Logic Mismatch: User specified '{day}', but date resolved to {actual_weekday} ({validated.departure_date})!")

    print("🟢 PASSED CUSTOMS INSPECTION:")
    # Avoids standard json.dumps TypeError by utilizing native Pydantic serialization
    print(validated.model_dump_json(indent=2))

except Exception as err:
    print(f"🔴 IMPOUNDED AT GATE 4: {err}")

```

---

## 📊 Empirical Evaluation: Schema Drift in Production

The matrix below illustrates the behavioral divide between prompt-only extraction and schema-gated agent extraction across multiple runs.

> **Empirical Benchmark Note:** Values reflect observed runs using openai/gpt-oss-120b. Because unconstrained LLMs are non-deterministic, exact strings and mutations will vary per run, while the systemic failure modes persist.

| Feature / Metric | Unstructured Run 1 | Unstructured Run 2 | Unstructured Run 3 | Structured Output (Cleared) | Production Failure Mode |
| --- | --- | --- | --- | --- | --- |
| **Origin Shape** | `dict` (`{"city": "...", "airport_code": "..."}`) | `str` (`"San Francisco (SFO)"`) | `str` (`"San Francisco (SFO)"`) | `str` (`"San Francisco"`) | `TypeError`: Downstream services calling `.upper()` or regex fail on nested mappings. |
| **Destination Value** | Nested mapping with airport code (`NRT`) | String with note: `"Tokyo (any airport)"` | String with note: `"Tokyo (any airport)"` | Normalized literal: `"Tokyo"` | Foreign key lookup mismatch in downstream airport databases. |
| **Date Key Name** | `"departure_date"` | `"departure_date"` | Renamed: `"date"` | Guaranteed: `"departure_date"` | `KeyError: 'departure_date'` halts asynchronous message queue workers. |
| **Computed Date** | `2026-09-27` *(Sunday)* | `2026-09-26` *(Saturday)* | `2026-09-27` *(Sunday)* | `2026-09-25` *(Friday)* | **Business Disruption:** Miscomputed travel dates cause missed customer meetings. |
| **Budget Key** | Renamed: `"budget_usd"` | Renamed: `"budget_usd"` | Renamed: `"budget_usd"` | Guaranteed: `"budget_limit_usd"` | ORM schema validation fails; payload rejected by payment APIs. |
| **Meal Key** | Renamed: `"meal_preference"` | Renamed: `"meal_preference"` | Renamed: `"meal_preference"` | Guaranteed: `"meal"` | Catering ingestion microservice discards invalid field names. |
| **Field Ordering** | Top-down | Top-down | Reordered (`origin` first, `name` last) | Canonical Schema Order | Breaks byte-level hashing, idempotency keys, and payload deduplication caches. |
| **Intent Routing** | Missing | Missing | Missing | Included: `"flight_booking"` | Message routers cannot classify the downstream destination queue. |

---

## 💡 Production Architecture Insights & Key Lessons

1. **Decouple Semantic Extraction from Mathematical Computation**
LLMs lack an internal clock and struggle with calendar calculations. Requesting an ISO-8601 date from phrases like `"next Friday"` without an anchored reference forces hallucinated calculations. Always provide an active reference clock in the system prompt (`Today is {datetime.now()}`).
2. **Prevent Native Date Serialization Exceptions**
Pydantic parses ISO strings directly into Python `datetime.date` objects. Passing `validated.model_dump()` to `json.dumps()` raises `TypeError: Object of type date is not JSON serializable`. Use Pydantic's built-in `validated.model_dump_json()` instead.
3. **Layer Deterministic Assertions at Gate 4**
While LLMs excel at fuzzy semantic parsing, they must not have final authority over strict invariants. Cross-reference the extracted user phrase against verified calendar logic (e.g., verifying `validated.departure_date.strftime("%A") == "Friday"`) before completing state changes.
4. **Bypass Fragile Notebook UI Event Loops**
In managed cloud environments (JupyterLab, Google Colab, VS Code Remote), widget callbacks (`ipywidgets.Button.on_click`) can fail silently when WebSocket comm channels disconnect. Using native IPython output flushing (`clear_output(wait=True)` and `display(HTML(...))`) ensures stable, reactive state rendering across all notebook environments.
