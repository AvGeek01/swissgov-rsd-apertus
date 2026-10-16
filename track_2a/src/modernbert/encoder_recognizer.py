from encoders.modeling_modernbert import ModernBertForTokenRegression
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

from rsd.recognizers.base import DifferenceRecognizer
from rsd.recognizers.utils import DifferenceSample
from encoders import utils


class EncoderDifferenceRecognizer(DifferenceRecognizer):
    def __init__(self, model_path_or_name: Union[str, Path]):
        self.model_path_or_name = model_path_or_name
        config = AutoConfig.from_pretrained(self.model_path_or_name, num_labels=1)
        if hasattr(config, "reference_compile"): 
            config.reference_compile = False
        if config.architectures and ("ModernBertForTokenRegression" in config.architectures):
            self.model = ModernBertForTokenRegression.from_pretrained(self.model_path_or_name, config=config, trust_remote_code=True)
        elif config.architectures and ("XLMRobertaXLForTokenClassification" in config.architectures or "XLMRobertaXLForTokenRegression" in config.architectures):
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path_or_name, num_labels=2, config=config, device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path_or_name, num_labels=1, config=config, device_map="auto", trust_remote_code=True)
        self.num_virtual_tokens = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_name, trust_remote_code=True)

    def __str__(self):
        return f"EncoderRecognizer(model_name={self.model_path_or_name})"

    def _predict(self, text_a: str, text_b: str):
        # Format input with dummy labels for text_b
        inputs = utils.format_as_sequence_pair(
            {
                "text_a": text_a,
                "text_b": text_b,
                "labels_b": [0] * len(text_b.split())
            },
            self.tokenizer
        )
        input_ids = inputs["input_ids"].unsqueeze(0)
        attention_mask = inputs["attention_mask"].unsqueeze(0)
        outputs = self.model(input_ids=input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device))
        logits = outputs.logits[0]
        if logits.size(-1) == 2:
            logits = logits[:, 0]
        if self.num_virtual_tokens > 0:
            logits = logits[self.num_virtual_tokens:]
        assert logits.size(0) == input_ids.size(1)

        predictions = torch.sigmoid(logits)

        # Identify where the second sequence starts
        tokens = input_ids[0].tolist()
        special_positions = [
            i for i, token_id in enumerate(tokens)
            if token_id in self.tokenizer.all_special_ids
        ]

        if len(special_positions) < 2:
            raise ValueError("Could not find second special token in input_ids")
        seq_b_start = special_positions[1] + 1

        # Collect offsets for alignment
        tokenized = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        offsets = tokenized["offset_mapping"][0].tolist()

        # Prepare for word-level aggregation
        words = text_b.split()
        word_spans = []
        running_char_index = 0
        for idx, w in enumerate(words):
            word_start = running_char_index if idx == 0 else running_char_index - 1
            word_end = running_char_index + len(w)
            word_spans.append((word_start, word_end, idx))
            running_char_index = word_end + 1

        # Distribute token-level predictions into word-level bins using offsets
        word_predictions = [[] for _ in words]
        for i in range(seq_b_start, len(tokens)):
            if tokens[i] in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                continue
            start_char, end_char = offsets[i]
            if start_char == 0 and end_char == 0:
                continue
            for (w_start, w_end, w_idx) in word_spans:
                if w_start <= start_char < w_end:
                    word_predictions[w_idx].append(predictions[i].item())
                    break

        # Average subword predictions for each word
        labels = []
        for sublist in word_predictions:
            if sublist:
                labels.append(sum(sublist) / len(sublist))
            else:
                labels.append(0.0)

        assert len(words) == len(labels)
        return words, labels

    @torch.no_grad()
    def predict(self, a: str, b: str, **kwargs) -> DifferenceSample:
        tokens_a, labels_a = self._predict(b, a)  # predictions for text A
        tokens_b, labels_b = self._predict(a, b)  # predictions for text B
        return DifferenceSample(
            tokens_a=tuple(tokens_a),
            tokens_b=tuple(tokens_b),
            labels_a=tuple(labels_a),
            labels_b=tuple(labels_b)
        )
