from pathlib import Path

import jsonlines
from tqdm import tqdm

from encoders.encoder_recognizer import EncoderDifferenceRecognizer
from evaluation.utils import load_gold_data
from rsd.recognizers import DiffAlign
from rsd.recognizers.base import DifferenceRecognizer

import torch


def main(recognizer: DifferenceRecognizer):


    def predict_and_save(recognizer: DifferenceRecognizer, gold_samples, gold_path):

        # Read sample IDs from the gold file
        with jsonlines.open(gold_path) as f:
            sample_ids = [line['id'] for line in f]

        if args.test_data == 'swissgov':
            if args.split == 'val':
                out_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'encoder_predictions' / 'dev' / args.split
            else:
                out_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'encoder_predictions' / args.split
        else:
            out_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'encoder_predictions' / 'full'
        out_dir.mkdir(parents=True, exist_ok=True)

        if "admin" in gold_path.name:
            suffix = gold_path.name.split('_')[-2:]
            if args.short and "_short" not in gold_path.name:
                out_path = out_dir / f"{str(recognizer).replace('/', '_')}_{suffix[0]}_{suffix[1]}_short.jsonl"
            else:
                out_path = out_dir / f"{str(recognizer).replace('/', '_')}_{suffix[0]}_{suffix[1]}.jsonl"
        else:
            out_path = out_dir / f"{str(recognizer).replace('/', '_')}.jsonl"
        print(f"Saving predictions to {out_path}")

        counter = 0
        with jsonlines.open(out_path, 'w') as f:
            for sample, id in tqdm(list(zip(gold_samples, sample_ids))):

                prediction = recognizer.predict(
                    a=" ".join(sample.tokens_a),
                    b=" ".join(sample.tokens_b),
                )
                assert prediction.tokens_a == sample.tokens_a
                assert prediction.tokens_b == sample.tokens_b
                ### comment this out when not working with rsd xlm-r or gte-multilingual-base or bge-m3
                if "roberta" in str(recognizer) or "gte" in str(recognizer).lower() or "bge-m3" in str(recognizer) or "MEXMA" in str(recognizer) or "sentence-transformers" in str(recognizer) or "gemma" in str(recognizer) or "eurobert" in str(recognizer).lower():
                    if len(prediction.labels_a) < len(sample.labels_a):
                            # add padding to prediction.labels_a
                            prediction.labels_a = list(prediction.labels_a) + [0.] * (len(sample.labels_a) - len(prediction.labels_a))
                            counter += 1
                    elif len(prediction.labels_a) > len(sample.labels_a):
                        # truncate prediction.labels_a
                        prediction.labels_a = prediction.labels_a[:len(sample.labels_a)]
                        counter += 1
                ###
                assert len(prediction.labels_a) == len(sample.labels_a), f"{len(prediction.labels_a)} != {len(sample.labels_a)} for SAMPLE {sample} and PREDICTION {prediction}"
                if not all([l == -1 for l in sample.labels_b]):
                    ### comment this out when not working with rsd xlm-r
                    if "roberta" in str(recognizer) or "gte" in str(recognizer).lower() or "bge-m3" in str(recognizer) or "MEXMA" in str(recognizer) or "sentence-transformers" in str(recognizer) or "gemma" in str(recognizer) or "eurobert" in str(recognizer).lower():
                        if len(prediction.labels_b) < len(sample.labels_b):
                            # add padding to prediction.labels_b
                            prediction.labels_b = list(prediction.labels_b) + [0.] * (len(sample.labels_b) - len(prediction.labels_b))
                            counter += 1
                        elif len(prediction.labels_b) > len(sample.labels_b):
                            # truncate prediction.labels_b
                            prediction.labels_b = prediction.labels_b[:len(sample.labels_b)]
                            counter += 1
                    ###
                    assert len(prediction.labels_b) == len(sample.labels_b), f"{len(prediction.labels_b)} != {len(sample.labels_b)} for {sample}"
                f.write({
                    "text_a": " ".join(prediction.tokens_a),
                    "text_b": " ".join(prediction.tokens_b),
                    "labels_a": [round(label, 5) for label in prediction.labels_a],
                    "labels_b": [round(label, 5) for label in prediction.labels_b],
                    "id": id,
                })
        print(f"Saved {len(gold_samples)} predictions to {out_path}")
        print(f"Number of samples with different label lengths: {counter}")

    if args.test_data == 'rsd':
        gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'full' / 'gold.jsonl' 
        gold_samples = load_gold_data(gold_path)

        predict_and_save(recognizer, gold_samples, gold_path)

    elif args.test_data == 'swissgov':
        if not args.lang:
            # Process languages sequentially to avoid memory issues
            for lang in ['de', 'fr', 'it']:
                print(f"Processing {lang}...")
                if args.split == 'val':
                    gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'dev' / 'val' / f'gold_admin_{lang}{"_short" if args.short else ""}.jsonl'
                else:
                    gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / args.split / f'gold_admin_{lang}{"_short" if args.short else ""}.jsonl'
                gold_samples = load_gold_data(gold_path)
                predict_and_save(recognizer, gold_samples, gold_path)
                
                # Clear GPU cache between languages to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"Cleared GPU cache after processing {lang}")
        else:
            if args.split == 'val':
                gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'dev' / 'val' / f'gold_admin_{args.lang}{"short" if args.short else ""}.jsonl'
            else:
                gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / args.split / f'gold_admin_{args.lang}.jsonl'
            gold_samples = load_gold_data(gold_path)
            predict_and_save(recognizer, gold_samples, gold_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name_or_path', type=Path)
    parser.add_argument('--type', type=str, choices=['finetuned', 'diffalign'])
    parser.add_argument('--test_data', type=str, choices=['rsd', 'swissgov'])
    parser.add_argument('--lang', type=str, choices=['de', 'fr', 'it'])
    parser.add_argument('--split', type=str, choices=['dev', 'test', 'val', 'full'])
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--layer', type=int, default=-1,
                        help='Hidden layer to extract features from (diffalign only, default: -1)')
    args = parser.parse_args()

    if args.type == 'finetuned':
        if args.layer != -1:
            raise ValueError("--layer is only supported for --type diffalign")
        recognizer = EncoderDifferenceRecognizer(args.model_name_or_path)
    elif args.type == 'diffalign':
        recognizer = DiffAlign(args.model_name_or_path, layer=args.layer)
    else:
        raise ValueError(f"Invalid recognizer type: {args.type}")

    main(recognizer)
