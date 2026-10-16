import argparse
from dataclasses import dataclass
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback,
)

from encoders.data_collator import DataCollatorForTokenRegression
from encoders.modeling_modernbert import ModernBertForTokenRegression
from encoders.finetuning.data import load_encoder_training_data


@dataclass
class ModelConfig:
    model_name_or_path: str
    model_revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


def main(training_args, model_args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token_id is not None:
            tokenizer.pad_token = tokenizer.sep_token

    ################
    # Dataset
    ################
    dataset = load_encoder_training_data(tokenizer)
    collator = DataCollatorForTokenRegression(tokenizer=tokenizer)

    ################
    # Training
    ################
    model = ModernBertForTokenRegression.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=model_args.torch_dtype,
        reference_compile=False
    ).to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (TrainingArguments, ModelConfig)
    parser = argparse.ArgumentParser()
    
    # Add arguments for TrainingArguments
    training_group = parser.add_argument_group("Training Arguments")
    for field in TrainingArguments.__dataclass_fields__.values():
        training_group.add_argument(f"--{field.name}", type=field.type, default=field.default)
    
    # Add arguments for ModelConfig
    model_group = parser.add_argument_group("Model Arguments")
    for field in ModelConfig.__dataclass_fields__.values():
        model_group.add_argument(f"--{field.name}", type=field.type, default=field.default)
    
    return parser


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelConfig))
    args = parser.parse_args()
    
    # Split args into training_args and model_args
    training_dict = {k: v for k, v in vars(args).items() if k in TrainingArguments.__dataclass_fields__}
    model_dict = {k: v for k, v in vars(args).items() if k in ModelConfig.__dataclass_fields__}
    
    training_args = TrainingArguments(**training_dict)
    model_args = ModelConfig(**model_dict)
    
    main(training_args, model_args)
