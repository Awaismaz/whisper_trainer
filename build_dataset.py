import os
import csv
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Turkmen Whisper Dataset Builder")
parser.add_argument("--online", action="store_true", help="Use OpenAI's online transcription API")
parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large or gpt-4o-transcribe)")
parser.add_argument("--wav_dir", type=str, default="dataset/wavs", help="Directory containing WAV files")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output metadata CSV file (optional)")
args = parser.parse_args()

# === Auto-increment Metadata File ===
def get_incremented_csv_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

# === Paths ===
os.makedirs(args.wav_dir, exist_ok=True)
default_csv = os.path.join("dataset", "metadata.csv")
output_csv = args.output_csv if args.output_csv else get_incremented_csv_name(default_csv)

# === Online Mode ===
if args.online:
    load_dotenv(override=True)
    client = OpenAI()
    print(f"Using OpenAI online model: {args.model}")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["file", "text"])

        for fname in os.listdir(args.wav_dir):
            if fname.endswith(".wav"):
                filepath = os.path.join(args.wav_dir, fname)
                print(f"Transcribing (online): {fname}")
                with open(filepath, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file
                    )
                    writer.writerow([fname, transcription.text])

# === Offline Mode ===
else:
    import whisper
    print(f"Loading local Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["file", "text"])

        for fname in os.listdir(args.wav_dir):
            if fname.endswith(".wav"):
                filepath = os.path.join(args.wav_dir, fname)
                print(f"Transcribing (offline): {fname}")
                result = model.transcribe(filepath, language="tk", fp16=False)
                writer.writerow([fname, result["text"]])

print("✅ Transcription complete. Output saved to:", output_csv)
