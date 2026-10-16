from pathlib import Path

# pyrefly: ignore [missing-import]
from datasets import load_dataset
from encoders import utils


def load_encoder_training_data(tokenizer):
    encoder_inputs_dir = Path(__file__).parent.parent.parent / "data" / "finetuning" / "encoder_inputs"
    assert encoder_inputs_dir.exists(), f"Encoder inputs directory not found at {encoder_inputs_dir}"
    dataset = load_dataset("json", data_files={
        "train": str(encoder_inputs_dir / 'train_en.jsonl'),
        "eval": str(encoder_inputs_dir / 'valid_en.jsonl'),
    })
    dataset = dataset.map(
        lambda examples: utils.format_as_sequence_pair(
            {"text_a": examples["text_a"], "text_b": examples["text_b"], "labels_b": examples["labels_b"]},
            tokenizer,
        ),
        batched=False
    )
    return dataset
