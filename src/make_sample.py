import os
import random
from pathlib import Path
import pandas as pd
from openai import OpenAI
import time
from typing import Optional
from dotenv import load_dotenv

from API_KEY import OPENAI_API_KEY


labels = ["Routine", "Safety", "Urgency", "Distress"]


vessels = [
    "SEA BREEZE",
    "BLUE HORIZON",
    "GALILEE STAR",
    "MEDITERRANEAN DREAM",
    "RED SEA RUNNER",
    "COASTAL SPIRIT",
]

call_signs = [
    "4XAB1",
    "4XCD2",
    "4XEF3",
    "4XGH4",
    "4XJK5",
    "4XLM6",
]


locations_formal = [
    "32°49.0'N 034°59.0'E (Haifa area)",
    "31°48.0'N 034°38.0'E (Ashdod area)",
    "32°04.0'N 034°46.0'E (Tel Aviv area)",
    "32°10.0'N 034°48.0'E (Herzliya area)",
    "32°28.0'N 034°52.0'E (Hadera area)",
    "31°55.0'N 034°41.0'E (Palmahim area)",
    "32°19.0'N 034°51.0'E (Netanya area)",
    "29°33.0'N 034°57.0'E (Eilat area)",
]


locations_informal = [
    "5 nm west of Haifa",
    "Ashdod Port",
    "5 nm west of Tel Aviv",
    "10 nm offshore of Herzliya Marina",
    "Hadera power station area",
    "Palmahim Beach",
    "Netanya coastline",
    "Eilat Marina",
]

weather = [
    "rough seas",
    "strong wind",
    "heavy rain",
    "poor visibility",
    "fog",
    "choppy waves",
]

pob_choices = [1, 2, 3, 4, 5, 6]

scenarios_by_label = {
    "Distress": ["taking_on_water", "person_overboard", "fire_onboard"],
    "Urgency": ["engine_failure", "drifting", "medical_issue"],
    "Safety": ["floating_debris", "storm_warning", "navigation_hazard"],
    "Routine": ["position_report", "tow_request", "radio_check"],
}

codeword_by_label = {
    "Distress": "MAYDAY",
    "Urgency": "PAN PAN",
    "Safety": "SECURITE",
    "Routine": "NONE",
}

styles = ["formal", "informal", "third_party"]


def generate_spec(label: str) -> dict:
    """Generate a structured sample specification (ground truth + metadata).

    This spec is later used to produce `text` via templates (and later, via an LLM).
    """

    # If you enforce that "formal" messages must always include a codeword,
    # avoid generating Routine samples with formal style (Routine has no standard codeword).
    if label == "Routine":
        rand_style = random.choice(["informal", "third_party"])
    else:
        rand_style = random.choice(styles)
    # Pick a style first, then decide which fields may be missing (especially for informal stress/non-compliance).

    # Informal messages may omit details due to stress (probabilistic missingness).
    # In informal calls, details are often missing - we simulate that with a simple omission probability.
    informal_omit_p = 0.6  # 60% omit, 40% include
    omit_pob_in_informal = (rand_style == "informal") and (
        random.random() < informal_omit_p
    )
    omit_weather_in_informal = (rand_style == "informal") and (
        random.random() < informal_omit_p
    )
    omit_vessel_in_informal = (rand_style == "informal") and (
        random.random() < informal_omit_p
    )
    omit_location_in_informal = (rand_style == "informal") and (
        random.random() < informal_omit_p
    )
    omit_callsign_in_informal = (rand_style == "informal") and (
        random.random() < informal_omit_p
    )
    omit_mmsi_in_informal = (rand_style == "informal") and (
        random.random() < informal_omit_p
    )

    rand_vessel = "" if omit_vessel_in_informal else random.choice(vessels)
    rand_call_sign = "" if omit_callsign_in_informal else random.choice(call_signs)
    rand_mmsi = (
        "" if omit_mmsi_in_informal else str(random.randint(200000000, 799999999))
    )

    rand_weather = "" if omit_weather_in_informal else random.choice(weather)

    # People on board (POB): for third-party reports this is often unknown.
    # Third-party observers usually don't know POB, so we mark it as unknown instead of guessing.
    if rand_style == "third_party":
        pob = "POB UNKNOWN"
    else:
        if omit_pob_in_informal:
            pob = ""
        else:
            rand_pob = random.choice(pob_choices)
            pob = (
                f"{rand_pob} PERSON ON BOARD"
                if rand_pob == 1
                else f"{rand_pob} PERSONS ON BOARD"
            )

    if rand_style == "formal":
        rand_location = random.choice(locations_formal)
    else:
        rand_location = (
            "" if omit_location_in_informal else random.choice(locations_informal)
        )

    # Without at least one anchor, the sample becomes too vague to be useful for analysis/training.
    # Safeguard: in informal messages, keep at least one anchor (vessel or location)
    if rand_style == "informal" and rand_vessel == "" and rand_location == "":
        rand_location = random.choice(locations_informal)

    rand_scenario_type = random.choice(scenarios_by_label[label])

    # Codewords follow radio procedure more strictly in formal style; elsewhere we keep it probabilistic.
    # Decide codeword independently from templates (important for future LLM generation)
    if label == "Routine":
        has_codeword = False
    elif rand_style == "formal":
        has_codeword = True
    else:
        has_codeword = random.random() < 0.6

    if has_codeword:
        codeword_type = codeword_by_label[label]
        # Third-party messages typically relay/hear the distress/urgency broadcast.
        if rand_style == "third_party" and codeword_type != "NONE":
            codeword = f"{codeword_type} RELAY"
        else:
            codeword = codeword_type
    else:
        codeword_type = "NONE"
        codeword = ""

    spec = {
        "label": label,
        "scenario_type": rand_scenario_type,
        "style": rand_style,
        "vessel": rand_vessel,
        "call_sign": rand_call_sign,
        "mmsi": rand_mmsi,
        "location": rand_location,
        "weather": rand_weather,
        "pob": pob,
        "has_codeword": has_codeword,
        "codeword_type": codeword_type,
        "codeword": codeword,
    }

    return spec


def build_llm_prompt(spec: dict) -> str:
    """Build a single prompt for an LLM to generate ONE maritime SAR/VHF message.

    The prompt is assembled from four parts:
    1) Global constraints (no hallucinations, use spec values exactly, return text only),
    2) Style-specific rules (formal / informal / third_party),
    3) The spec block (ground truth fields like scenario_type, vessel, location, pob, weather, codeword),
    4) A final reminder to output only the message text.

    This function returns the prompt string that will later be sent to the LLM to generate the `text`
    while keeping the output consistent with the structured spec (and allowing missing fields for informal style).
    """
    # Prompt pieces: global constraints + style rules + ground-truth spec -> one clear instruction to the LLM.

    # Per-style writing/formatting rules (the LLM should follow these for tone + structure).
    style_rules = {
        "formal": [
            "STRICT FORMAT REQUIRED: Follow the standard MIP structure exactly.",
            "1. START: Repeat the codeword 3 times (e.g., 'MAYDAY, MAYDAY, MAYDAY').",
            "2. IDENTITY: Say 'THIS IS' followed by the vessel name repeated 3 times.",
            "3. DETAILS: Mention Call Sign and MMSI once immediately after the name.",
            "4. POSITION: Clearly state the location/coordinates.",
            "5. NATURE: Describe the distress scenario clearly.",
            "6. ASSISTANCE: State 'REQUIRE IMMEDIATE ASSISTANCE'.",
            "7. POB: State number of persons on board.",
            "8. END: Finish with 'OVER.'",
            "Use uppercase only.",
            "Do NOT add pleasantries like 'Hello' or 'Please'.",
        ],
        "informal": [
            "Write in a panicked or stressed informal radio style.",
            "You may omit protocol formalities (like repeating names).",
            "Focus on the urgency and the problem.",
            "Natural casing (not all caps).",
            "You might forget to say 'OVER' or identify yourself clearly.",
        ],
        "third_party": [
            "Write as a third-party observer relaying a message.",
            "IDENTITY: You are NOT the vessel in the spec. Choose a random name for yourself.",
            "TARGET: The 'vessel' in the spec is the source of the distress.",
            "START: You MUST start by repeating the codeword from the spec followed by the word 'RELAY' 3 times (e.g. 'MAYDAY RELAY, MAYDAY RELAY, MAYDAY RELAY').",
            "CONTENT: State clearly: 'RECEIVED [CODEWORD] FROM [vessel name]'. (Do not add 'RELAY' here, as you received the original distress signal).",
            "Report the situation and position provided in the spec.",
            "End with 'OVER'.",
        ],
    }

    style = spec.get("style", "informal")
    selected_rules = style_rules.get(style, style_rules["informal"])
    style_block = "\n".join(f"- {s}" for s in selected_rules)

    # Global grounding rules to reduce hallucinations and keep outputs aligned with the spec.
    # Constraints: keep the LLM grounded in the structured spec.
    constraints = [
        "Do not invent any facts beyond what is provided in the spec.",
        "Use the exact location string as given (do not rephrase or shorten it).",
        "The message must clearly describe the scenario_type from the spec.",
        "If codeword is empty, do not add MAYDAY / PAN PAN / SECURITE or any other codeword.",
        "Return only the message text (no JSON, no explanations).",
    ]

    # Constraints depend on style because informal samples may intentionally omit some fields.
    # Style-specific grounding.
    if style == "formal":
        constraints.extend(
            [
                "You must mention the vessel name.",
                "Structure MUST be: MAYDAY x3 -> THIS IS NAME x3 -> Position -> Distress -> Assistance -> POB -> OVER.",  # חידוד הסדר
                "You must mention the number of people on board using the pob value from the spec.",
                "You must mention the weather only if it is relevant to the distress, otherwise keep it brief.",
                "You must include the codeword from the spec.",
            ]
        )
    elif style == "informal":
        constraints.extend(
            [
                "If vessel is empty, do not invent a vessel name. If vessel is provided, use it exactly.",
                "If location is empty, do not invent a location. If location is provided, use it exactly.",
                "If call_sign is empty, do not invent a call sign. If call_sign is provided, use it exactly.",
                "If mmsi is empty, do not invent an MMSI. If mmsi is provided, use it exactly.",
                "If pob is empty or 'POB UNKNOWN', do not provide a number of people on board. If pob is provided, use it exactly.",
                "If weather is empty, do not add weather details. If weather is provided, use it exactly.",
            ]
        )
    else:
        constraints.extend(
            [
                "You must mention the vessel name.",
                "Do not invent the number of people on board. If pob is not 'POB UNKNOWN', you may mention it; otherwise do not provide a number.",
                "You must mention the weather.",
            ]
        )

    constraints_block = "\n".join(f"- {c}" for c in constraints)

    # Serialize the spec into a readable block the model must stick to (no rephrasing of values).
    spec_lines = [
        f"scenario_type: {spec['scenario_type']}",
        f"style: {style}",
        f"vessel: {spec['vessel']}",
        f"call_sign: {spec['call_sign']}",
        f"mmsi: {spec['mmsi']}",
        f"location: {spec['location']}",
        f"pob: {spec['pob']}",
        f"weather: {spec['weather']}",
        f"codeword: {spec['codeword']}",
    ]
    spec_block = "\n".join(spec_lines)

    # Assemble the final prompt in a clean sectioned format (models follow structure better).
    prompt_lines = [
        "You are generating ONE synthetic maritime SAR/VHF message. Follow the constraints and the style rules, and use the spec values exactly.",
        "CONSTRAINTS:\n" + constraints_block,
        "STYLE RULES:\n" + style_block,
        "SPEC (ground truth):\n" + spec_block,
        "Return only the message text.",
    ]
    lines_block = "\n\n".join(prompt_lines).strip()

    return lines_block


client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm(prompt: str) -> str:
    """Send ONE prompt to the OpenAI API and return only the generated message text."""

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        temperature=0.4,
        max_output_tokens=120,
    )

    text = (response.output_text).strip()
    return text


# --- LLM-based generation utilities ---


def generate_text_llm(spec: dict) -> str:
    """Generate the message text from a spec using the LLM (spec -> prompt -> text)."""
    prompt = build_llm_prompt(spec)
    return call_llm(prompt)


def spec_to_row(spec: dict, text: str) -> dict:
    """Flatten spec + generated text into a single tabular row."""
    row = dict(spec)
    row["text"] = text
    return row


def _safe_llm_call(
    spec: dict, max_retries: int = 4, base_sleep: float = 1.5
) -> Optional[str]:
    """Best-effort LLM call with small retries for transient failures / rate limits."""
    for attempt in range(max_retries):
        try:
            return generate_text_llm(spec)
        except Exception as e:
            # Keep it simple: exponential-ish backoff
            sleep_s = base_sleep * (2**attempt)
            print(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            print(f"Sleeping {sleep_s:.1f}s and retrying...")
            time.sleep(sleep_s)
    return None


def generate_dataset(
    n_per_label: int = 25,
    seed: int = 42,
    out_path: Optional[str] = None,
    save_every: int = 25,
) -> pd.DataFrame:
    """Generate a dataset by sampling specs and using the LLM to produce message text.

    - n_per_label: how many samples to generate for each label
    - seed: RNG seed for reproducibility
    - out_path: if provided, saves a CSV to this path (and incremental checkpoints)
    - save_every: write a checkpoint every N rows (only if out_path provided)
    """
    random.seed(seed)

    rows = []
    total = n_per_label * len(labels)
    produced = 0

    for label in labels:
        for _ in range(n_per_label):
            spec = generate_spec(label)
            text = _safe_llm_call(spec)

            # If the LLM call failed after retries, keep a row with empty text so you can filter later.
            if text is None:
                text = ""

            rows.append(spec_to_row(spec, text))
            produced += 1

            if produced % 10 == 0 or produced == total:
                print(f"Progress: {produced}/{total}")

            if out_path and (produced % save_every == 0 or produced == total):
                df_ckpt = pd.DataFrame(rows)
                df_ckpt.to_csv(out_path, index=False, encoding="utf-8")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Quick sanity: print one prompt + one generated sample
    spec = generate_spec(random.choice(labels))
    print("--- PROMPT ---")
    print(build_llm_prompt(spec))
    print("\n--- GENERATED TEXT ---")
    print(generate_text_llm(spec))


df = generate_dataset(n_per_label=75, seed=42, out_path="seaalert_samples.csv")
print(df.head())
