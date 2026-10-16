# Methodology: SwissGov-RSD Track 2A

## Problem
Detect token-level semantic differences between parallel Swiss government
documents in English and German, French, or Italian.

## Approach
The experiment uses the Apertus 8B Instruct model family. The workflow
includes preparation of labelled training examples, prompt-based inference,
a QLoRA fine-tuning experiment, and validation of generated token-level
scores.

## Fine-tuning
A QLoRA run was completed for 20 training steps using a 4-bit quantized
base model. The experiment used a maximum sequence length of 1024 tokens.

## Validation status
Training completed, but the adapter has not yet passed reliable output
validation. Initial outputs showed repetitive scores and formatting or
token-count mismatches. No performance improvement is claimed.

## Reproducibility
The final submission must document model revision, dependencies, data
provenance, configuration, hardware, and exact execution commands.

## Limitations
The short training run and unresolved output validation prevent reliable
claims about predictive quality. Additional evaluation is required.
