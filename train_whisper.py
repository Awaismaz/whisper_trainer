import os
import argparse
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper model on Turkmen dataset")
    parser.add_argument('--model', default='openai/whisper-tiny', help='Hugging Face model ID (default: openai/whisper-small)')
    parser.add_argument('--csv', default='dataset/metadata.csv', help='Path to metadata CSV file (default: dataset/metadata.csv)')
    parser.add_argument('--audio_dir', default='dataset/wavs', help='Directory containing audio files (default: dataset/wavs)')
    parser.add_argument('--output_dir', default='./whisper-turkmen-checkpoints', help='Directory to save checkpoints (default: ./whisper-turkmen-checkpoints)')
    parser.add_argument('--max_steps', type=int, default=5000, help='Maximum number of training steps (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size per device (default: 4)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load CSV and attach full file paths
    df = pd.read_csv(args.csv)
    df["file"] = df["file"].apply(lambda x: os.path.join(args.audio_dir, x))
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("file", Audio(sampling_rate=16000))


    # Load model and processor
    processor = WhisperProcessor.from_pretrained(args.model, language="turkmen", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    def data_collator(batch):
        input_features = [example["input_features"] for example in batch]
        label_features = [example["labels"] for example in batch]

        batch_input = processor.feature_extractor.pad(
            {"input_features": input_features},
            return_tensors="pt"
        )

        batch_labels = processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt",
            padding=True
        )

        # Replace padding token id with -100 so it's ignored in loss
        batch_labels["input_ids"][batch_labels["input_ids"] == processor.tokenizer.pad_token_id] = -100

        batch_input["labels"] = batch_labels["input_ids"]
        return batch_input


    # Preprocess
    def preprocess(batch):
        audio = batch["file"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(preprocess)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=500,
        # evaluation_strategy="no",
        fp16=True,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator
    )
    print("🖥️ Device being used:", next(model.parameters()).device)

    # Train
    trainer.train()

if __name__ == "__main__":
    main()
