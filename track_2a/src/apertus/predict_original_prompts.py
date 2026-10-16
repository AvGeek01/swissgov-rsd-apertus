import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
INPUT_PATH = Path("data/evaluation/llm_inputs/test_admin_de.jsonl")
SOURCE_PATH = Path(
    "data/evaluation/llm_predictions/full/"
    "gpt-4o-2024-08-06_admin_de.jsonl"
)
OUTPUT_PATH = Path(
    "data/evaluation/llm_predictions/"
    "apertus_original_prompts/apertus8b_admin_de_scores.jsonl"
)

CHUNK_SIZE = 10
VALID_SCORES = {-1, 0, 1, 2, 3, 4, 5}
CODE_TO_SCORE = {
    "X": -1,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


def get_tokens(prompt):
    marker = "## Input to Annotate"
    if marker not in prompt:
        raise ValueError("Input section not found")

    actual = prompt.split(marker, 1)[1]
    match = re.search(
        r"Sentence 1:\s*(\[.*?\])\s*"
        r"Sentence 2:\s*(\[.*?\])",
        actual,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not extract sentence token lists")

    return json.loads(match.group(1)), json.loads(match.group(2))


def get_scoring_instructions(prompt):
    start = prompt.find("## Score scale")
    end = prompt.find("## Input to Annotate")

    if start == -1 or end == -1 or start >= end:
        raise ValueError("Could not extract original scoring instructions")

    return prompt[start:end].strip()


def score_chunk(model, tokenizer, target, context, side, instructions):
    prompt = f"""
You are annotating semantic correspondence between an English sentence
and its German translation.

Follow these scoring instructions:
{instructions}

Score each target token according to its meaning in the context of
the other sentence.

Target side: {side}
Target tokens:
{json.dumps(target, ensure_ascii=False)}

Other sentence for context:
{json.dumps(context, ensure_ascii=False)}

Scoring rules:
- 5: complete semantic equivalence
- 4: very close correspondence
- 3: substantial but imperfect correspondence
- 2: some semantic similarity
- 1: slight semantic similarity
- 0: no semantic correspondence
- -1: punctuation tokens only

Important:
- Produce exactly {len(target)} integers.
- Each integer corresponds to the target token at the same position.
- Score each token independently; do not copy a repeating pattern.
- Never assign -1 to a word.
- Do not include token names or explanations.

Output format: one integer per line, with no bullets or Markdown.
""".strip()

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

    # Accept plain integer lines, optionally inside a code block.
    cleaned = re.sub(
        r"```(?:json|text)?", "", response, flags=re.IGNORECASE
    )
    cleaned = cleaned.replace("```", "").strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    if len(lines) == len(target):
        try:
            scores = [int(line) for line in lines]
        except ValueError:
            scores = []

        if (
            len(scores) == len(target)
            and all(score in VALID_SCORES for score in scores)
        ):
            return scores

    raise ValueError(
        f"{side} chunk expected a JSON array of {len(target)} valid scores. "
        f"Model response: {response[:500]!r}"
    )

def score_sentence(model, tokenizer, target, context, side, instructions):
    scores = []

    for start in range(0, len(target), CHUNK_SIZE):
        chunk = target[start:start + CHUNK_SIZE]
        end = start + len(chunk)

        print(f"  {side}: tokens {start + 1}-{end}/{len(target)}")

        chunk_scores = score_chunk(
            model,
            tokenizer,
            chunk,
            context,
            side,
            instructions,
        )

        if len(chunk_scores) != len(chunk):
            raise ValueError(f"Score count mismatch in {side}, chunk {start}")

        scores.extend(chunk_scores)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    with INPUT_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:args.limit]

    with SOURCE_PATH.open(encoding="utf-8") as f:
        source_rows = [
            json.loads(line) for line in f if line.strip()
        ][:args.limit]

    if len(rows) != len(source_rows):
        raise ValueError("Input/source record count mismatch")

    for i, (row, source) in enumerate(zip(rows, source_rows)):
        reconstructed = row["messages"][0]["content"]
        original = source["api_request"]["body"]["messages"][0]["content"]

        if reconstructed != original:
            raise ValueError(f"Original prompt mismatch at record {i}")

    print("Original prompts verified.")
    print(f"Records to process: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append mode lets us retain completed records if a later record fails.
    # For this first smoke test, use a fresh output file.
    if args.limit == 1 and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for index, (row, source) in enumerate(
            zip(rows, source_rows), start=1
        ):
            prompt = row["messages"][0]["content"]
            tokens_a, tokens_b = get_tokens(prompt)
            instructions = get_scoring_instructions(prompt)
            item_id = source.get("item_id", f"record_{index - 1}")

            print(
                f"\nRecord {index}/{len(rows)}: {item_id} "
                f"({len(tokens_a)} English, {len(tokens_b)} German tokens)"
            )

            scores_a = score_sentence(
                model, tokenizer, tokens_a, tokens_b,
                "English", instructions
            )
            scores_b = score_sentence(
                model, tokenizer, tokens_b, tokens_a,
                "German", instructions
            )

            if len(scores_a) != len(tokens_a):
                raise ValueError("English final score count mismatch")
            if len(scores_b) != len(tokens_b):
                raise ValueError("German final score count mismatch")

            prediction = {
                "sentence1": [
                    [token, score]
                    for token, score in zip(tokens_a, scores_a)
                ],
                "sentence2": [
                    [token, score]
                    for token, score in zip(tokens_b, scores_b)
                ],
            }

            # Final checks before saving.
            for key, expected in (
                ("sentence1", tokens_a),
                ("sentence2", tokens_b),
            ):
                pairs = prediction[key]
                if [pair[0] for pair in pairs] != expected:
                    raise ValueError(f"{key} token alignment failed")
                if any(pair[1] not in VALID_SCORES for pair in pairs):
                    raise ValueError(f"{key} contains invalid scores")

            out.write(json.dumps({
                "item_id": item_id,
                "prediction": prediction,
            }, ensure_ascii=False) + "\n")
            out.flush()

            print("Validation passed.")
            print("English score distribution:", {
                score: scores_a.count(score) for score in sorted(VALID_SCORES)
            })
            print("German score distribution:", {
                score: scores_b.count(score) for score in sorted(VALID_SCORES)
            })
            print("Saved:", OUTPUT_PATH)

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()
