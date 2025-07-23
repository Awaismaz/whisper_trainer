import os
import csv
import argparse
from dotenv import load_dotenv
from openai import OpenAI
# Parse CLI arguments
parser = argparse.ArgumentParser(description="Turkmen Whisper Dataset Builder")
parser.add_argument("--online", action="store_true", help="Use OpenAI's online transcription API")
parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large or gpt-4o-transcribe)")
args = parser.parse_args()

# Config
wav_dir = "dataset/wavs"
output_csv = "dataset/metadata.csv"

# Online Mode
if args.online:
    

    # Load API key
    load_dotenv(override=True)
    client = OpenAI()
    print(f"Using OpenAI online model: {args.model}")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["file", "text"])

        for fname in os.listdir(wav_dir):
            if fname.endswith(".wav"):
                filepath = os.path.join(wav_dir, fname)
                print(f"Transcribing (online): {fname}")
                with open(filepath, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file
                    )
                    writer.writerow([fname, transcription.text])

else:
    import whisper
    print(f"Loading local Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["file", "text"])

        for fname in os.listdir(wav_dir):
            if fname.endswith(".wav"):
                filepath = os.path.join(wav_dir, fname)
                print(f"Transcribing (offline): {fname}")
                result = model.transcribe(filepath, language="tk", fp16=False)
                writer.writerow([fname, result["text"]])

print("✅ Transcription complete. Output saved to:", output_csv)
