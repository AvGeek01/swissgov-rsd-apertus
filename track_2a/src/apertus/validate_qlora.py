import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.stats import spearmanr, kendalltau

MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
INPUT_PATH = Path("data/evaluation/gold_labels/dev/val/gold_admin_de.jsonl")
ADAPTER_PATH = Path("out/apertus_qlora_bf16_smoke")

# The training labels are reversed relative to the evaluation scale.
TO_EVAL = {
    -1: -1.0,
    0: 1.0,
    1: 0.8,
    2: 0.6,
    3: 0.4,
    4: 0.2,
    5: 0.0,
}
VALID_LABELS = set(TO_EVAL.keys())
VALID_SCORES = set(TO_EVAL.values())



def extract_json(response):
    response = re.sub(r"^(?:json)?\s*", "", response.strip(), flags=re.I)
    response = re.sub(r"\s*\s*$", "", response)

    start = response.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found. Response: {response[:500]!r}")

    response = response[start:]

    # First try the original output without modifications.
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Repair the specific malformed pattern:
    # "token", 0]  ->  ["token", 0]
    # Only applies when a quoted token is followed by an integer label
    # and a closing bracket, but is missing its opening bracket.
    repaired = re.sub(
        r'(?<!\[)"((?:\\.|[^"\\])*)"\s*,\s*(-?\d+)\s*\]',
        r'["\1", \2]',
        response,
    )

    try:
        result = json.loads(repaired)
        print("Warning: repaired malformed JSON token-label pairs.")
        return result
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON remains invalid after repair: {exc}. "
            f"Response near error: {repaired[max(0, exc.pos - 150):exc.pos + 150]!r}"
        ) from exc



def build_prompt(row):
    tokens_a = row["text_a"].split()
    tokens_b = row["text_b"].split()

    template = Path("prompt_templates/template.txt").read_text(
        encoding="utf-8"
    )

    prompt = template.replace(
        "{{ sentence1 }}",
        json.dumps(tokens_a, ensure_ascii=False),
    ).replace(
        "{{ sentence2 }}",
        json.dumps(tokens_b, ensure_ascii=False),
    )

    return prompt, tokens_a, tokens_b




def generate_pair_labels(model, tokenizer, prompt, tokens_a, tokens_b):
    chunk_size = 25

    def score_sentence(target_key, target_tokens, other_tokens):
        all_scores = []

        for start in range(0, len(target_tokens), chunk_size):
            chunk = target_tokens[start:start + chunk_size]

            if target_key == "sentence1":
                sentence1 = chunk
                sentence2 = other_tokens
                expected_key = "sentence1"
                other_key = "sentence2"
            else:
                sentence1 = other_tokens
                sentence2 = chunk
                expected_key = "sentence2"
                other_key = "sentence1"

            task = f"""
Annotate the semantic similarity of each token in the TARGET list
against the OTHER sentence.

Labels:
5 = complete equivalence
4 = very similar
3 = closely similar
2 = somewhat related
1 = slightly related
0 = no semantic relation
-1 = punctuation

TARGET tokens: {json.dumps(chunk, ensure_ascii=False)}
OTHER sentence: {json.dumps(other_tokens, ensure_ascii=False)}

Return ONLY a JSON array containing exactly {len(chunk)} integer labels,
one for each TARGET token in its original order.
Example format: [5, 0, 3, -1]
Allowed labels: -1, 0, 1, 2, 3, 4, 5.
No token strings, explanations, or Markdown.
"""


            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                chat_prompt, return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            generated_count = (
                output_ids.shape[1] - inputs["input_ids"].shape[1]
            )
            print(
                f"{target_key}, chunk {start // chunk_size + 1}: "
                f"{len(chunk)} tokens; generated {generated_count} tokens"
            )


            # Accept either the requested integer array or the token-label
            # object format used in the fine-tuning targets.
            try:
                decoder = json.JSONDecoder()
                starts = [i for i, ch in enumerate(response) if ch in "[{"]
                parsed = None
                for pos in starts:
                    try:
                        parsed, _ = decoder.raw_decode(response[pos:])
                        break
                    except json.JSONDecodeError:
                        continue
                if parsed is None:
                    raise ValueError("No valid JSON array or object found")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse model output. Response: {response[-500:]!r}"
                ) from exc

            if isinstance(parsed, dict):
                entries = parsed.get(expected_key)
                if not isinstance(entries, list):
                    raise ValueError(
                        f"JSON object has no list for {expected_key!r}. "
                        f"Keys: {list(parsed.keys())!r}"
                    )

                labels = []
                for entry in entries:
                    if (
                        isinstance(entry, (list, tuple))
                        and len(entry) >= 2
                    ):
                        labels.append(entry[1])
                    elif isinstance(entry, dict) and "label" in entry:
                        labels.append(entry["label"])
                    else:
                        raise ValueError(
                            f"Unexpected token-label entry: {entry!r}"
                        )
            elif isinstance(parsed, list):
                # Also accept a list of [token, label] pairs.
                if parsed and all(
                    isinstance(entry, (list, tuple)) and len(entry) >= 2
                    for entry in parsed
                ):
                    labels = [entry[1] for entry in parsed]
                else:
                    labels = parsed
            else:
                raise ValueError(
                    f"Expected a JSON array or object, got "
                    f"{type(parsed).__name__}: {response[-500:]!r}"
                )

            if len(labels) != len(chunk):
                if (
                    len(labels) > len(chunk)
                    and all(
                        isinstance(label, int)
                        and not isinstance(label, bool)
                        and label in VALID_LABELS
                        for label in labels
                    )
                ):
                    print(
                        f"WARNING: {target_key} returned {len(labels)} labels for "
                        f"{len(chunk)} tokens; truncating extras. Check model quality."
                    )
                    labels = labels[:len(chunk)]
                else:
                    raise ValueError(
                        f"Expected {len(chunk)} labels, got {len(labels)}. "
                        f"Response: {response[-500:]!r}"
                    )

            for token, label in zip(chunk, labels):
                if isinstance(label, bool) or not isinstance(label, int):
                    raise ValueError(f"Expected integer label, got {label!r}")
                if label not in VALID_LABELS:
                    raise ValueError(f"Unexpected label: {label!r}")
                all_scores.append(float(TO_EVAL[label]))

        return all_scores

    scores_a = score_sentence("sentence1", tokens_a, tokens_b)
    scores_b = score_sentence("sentence2", tokens_b, tokens_a)

    return scores_a, scores_b


def load_model(use_adapter):
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH) if use_adapter else MODEL_ID
    )
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
    if use_adapter:
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    return tokenizer, model


def main():
    global ADAPTER_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["base", "adapter"], required=True,
        help="Evaluate the base model or the smoke-test LoRA adapter."
    )
    parser.add_argument("--limit", type=int, default=34)
    parser.add_argument(
        "--record-id",
        type=str,
        default=None,
        help="Evaluate only the validation record with this ID.",
    )
    parser.add_argument("--adapter-path", type=Path, default=ADAPTER_PATH)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path for validation predictions."
    )
    args = parser.parse_args()
    adapter_path = args.adapter_path

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.model == "adapter" and not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Adapter weights not found in {adapter_path}")

    
    with INPUT_PATH.open(encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    if args.record_id:
        rows = [row for row in all_rows if row["id"] == args.record_id]
        if not rows:
            raise ValueError(f"Record ID not found: {args.record_id}")
    else:
        rows = all_rows[:args.limit]

    if not rows:
        raise ValueError("Validation file is empty")


    print(f"Mode: {args.model}")
    print(f"Validation examples: {len(rows)}")
    print("Loading model...")
    if args.model == "adapter":
        ADAPTER_PATH = adapter_path
    tokenizer, model = load_model(args.model == "adapter")

    gold_all, pred_all = [], []
    output_path = args.output or Path(
        f"out/validation_{args.model}_gold_admin_de.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".jsonl.tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as out:
            for i, row in enumerate(rows, 1):
                prompt, tokens_a, tokens_b = build_prompt(row)
                if len(tokens_a) != len(row["labels_a"]) or len(tokens_b) != len(row["labels_b"]):
                    raise ValueError(f"Gold token count mismatch: {row['id']}")
                
                print(f"[{i}/{len(rows)}] {row['id']}: scoring paired sentences...")
                scores_a, scores_b = generate_pair_labels(
                    model, tokenizer, prompt, tokens_a, tokens_b
                )
                   

                for name, scores, expected in (
                    ("A", scores_a, len(tokens_a)),
                    ("B", scores_b, len(tokens_b)),
                ):
                    if len(scores) != expected:
                        raise ValueError(
                            f"{row['id']} sentence {name}: got {len(scores)} "
                            f"scores, expected {expected}"
                        )
                    if any(
                        isinstance(s, bool)
                        or not isinstance(s, (int, float))
                        or float(s) not in VALID_SCORES
                        for s in scores
                    ):
                        raise ValueError(f"{row['id']} sentence {name}: invalid scores")

                # Save evaluator-scale scores in EncoderPrediction format.
                pred_a = [float(s) for s in scores_a]
                pred_b = [float(s) for s in scores_b]
                record = {
                    "id": row["id"],
                    "text_a": row["text_a"],
                    "text_b": row["text_b"],
                    "labels_a": pred_a,
                    "labels_b": pred_b,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                for gold, pred in ((row["labels_a"], pred_a), (row["labels_b"], pred_b)):
                    for g, p in zip(gold, pred):
                        if float(g) != -1.0:
                            gold_all.append(float(g))
                            pred_all.append(float(p))

                print(
                    f"    tokens A={len(tokens_a)}, B={len(tokens_b)}; "
                    f"valid scored tokens so far={len(gold_all)}"
                )

        # Publish the completed result only after every row succeeds.
        temp_path.replace(output_path)
    except Exception:
        print(f"Validation did not complete. Partial output, if any: {temp_path}")
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gold = np.asarray(gold_all, dtype=float)
    pred = np.asarray(pred_all, dtype=float)
    print("\n=== Validation metrics (punctuation excluded) ===")
    print(f"Scored tokens: {len(gold)}")
    print(f"Gold mean: {gold.mean():.4f}; prediction mean: {pred.mean():.4f}")
    print(f"Prediction score counts: {dict(zip(*np.unique(pred, return_counts=True)))}")
    print(f"Spearman: {spearmanr(gold, pred).statistic:.4f}")
    print(f"Kendall tau: {kendalltau(gold, pred).statistic:.4f}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
