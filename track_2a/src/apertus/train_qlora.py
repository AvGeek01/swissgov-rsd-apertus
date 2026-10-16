import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
DATA_PATH = Path("data/finetuning/llm_inputs/apertus_train.jsonl")


def encode_example(row, tokenizer, max_length):
    messages = row["messages"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=False
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=False
    )["input_ids"]

    prefix_len = 0
    for a, b in zip(full_ids, prompt_ids):
        if a != b:
            break
        prefix_len += 1

    if len(full_ids) > max_length or prefix_len >= len(full_ids):
        return None

    labels = full_ids.copy()
    labels[:prefix_len] = [-100] * prefix_len

    if all(x == -100 for x in labels):
        return None

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def save_checkpoint(model, tokenizer, optimizer, output_dir, completed_steps):
    checkpoint_dir = output_dir / f"checkpoint-{completed_steps}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    torch.save(
        {
            "completed_steps": completed_steps,
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        checkpoint_dir / "training_state.pt",
    )

    print(f"Checkpoint saved: {checkpoint_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", default="out/apertus_qlora_smoke")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    if args.steps < 1 or args.save_every < 1:
        raise ValueError("--steps and --save-every must be positive integers.")

    if args.smoke and args.resume_from:
        raise ValueError("Do not combine --smoke with --resume-from.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the correct environment.")

    output_dir = Path(args.output)
    resume_dir = Path(args.resume_from) if args.resume_from else None

    if resume_dir:
        state_path = resume_dir / "training_state.pt"
        adapter_path = resume_dir / "adapter_config.json"
        if not state_path.is_file() or not adapter_path.is_file():
            raise FileNotFoundError(
                "Resume folder must contain training_state.pt and adapter_config.json."
            )

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Preparing examples without truncating targets...", flush=True)
    examples = []
    data_path=Path(args.data)
    print("Training data:", data_path, flush=True)
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            item = encode_example(row, tokenizer, args.max_length)
            if item is not None:
                examples.append(item)

    print(f"Examples fitting {args.max_length} tokens: {len(examples)}", flush=True)
    if len(examples) < 4:
        raise RuntimeError("Too few examples fit. Increase --max-length cautiously.")

    random.Random(42).shuffle(examples)

    if args.smoke:
        examples = examples[:4]
        print("SMOKE TEST: using 4 examples only.", flush=True)

    print("Loading Apertus in 4-bit BF16...", flush=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )

    if resume_dir:
        print(f"Loading adapter from {resume_dir}...", flush=True)
        model = PeftModel.from_pretrained(
            base_model, resume_dir, is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
    )

    start_step = 0
    if resume_dir:
        print("Restoring optimizer and training state...", flush=True)
        state = torch.load(
            resume_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["completed_steps"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming after step {start_step}.", flush=True)

    steps = min(args.steps, len(examples))
    if start_step >= steps:
        raise ValueError(
            f"Checkpoint already completed {start_step} steps; "
            f"--steps {steps} does not request any additional work."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    print(
        f"Training steps {start_step + 1} through {steps}; "
        f"checkpoint interval={args.save_every}.",
        flush=True,
    )

    for step_idx in range(start_step, steps):
        item = examples[step_idx]
        device = model.get_input_embeddings().weight.device
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        labels = item["labels"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        optimizer.zero_grad(set_to_none=True)
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = result.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step_idx + 1}: {loss.item()}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        completed_steps = step_idx + 1
        print(
            f"Step {completed_steps}/{steps} | "
            f"loss={loss.item():.4f} | tokens={input_ids.shape[1]}",
            flush=True,
        )

        del result, loss, input_ids, labels, attention_mask

        if (
            completed_steps % args.save_every == 0
            or completed_steps == steps
        ):
            save_checkpoint(
                model, tokenizer, optimizer, output_dir, completed_steps
            )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Final adapter saved to:", output_dir, flush=True)
    print(
        "Peak GPU allocated (GB):",
        round(torch.cuda.max_memory_allocated() / 1e9, 2),
        flush=True,
    )


if __name__ == "__main__":
    main()
