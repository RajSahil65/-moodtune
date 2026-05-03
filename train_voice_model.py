"""
train_voice_model.py — Train the voice emotion SVM from scratch

This script trains a Support Vector Machine on audio files organized
by emotion, then saves the model to backend/models/voice_svm.pkl

Dataset expected structure:
  dataset/
    happy/   → *.wav files
    sad/     → *.wav files
    angry/   → *.wav files
    neutral/ → *.wav files
    fear/    → *.wav files
    surprise/→ *.wav files

Recommended datasets:
  - RAVDESS: https://zenodo.org/record/1188976
  - CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
  - SAVEE:   http://kahlan.eps.surrey.ac.uk/savee/

Usage:
  python train_voice_model.py --dataset ./dataset --output backend/models/voice_svm.pkl
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Train voice emotion SVM")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", default="backend/models/voice_svm.pkl",
                        help="Output path for the model pickle")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"ERROR: Dataset directory not found: {args.dataset}")
        sys.exit(1)

    print(f"Loading training data from: {args.dataset}")
    sys.path.insert(0, "backend")

    from models.emotion_voice import train_voice_svm
    train_voice_svm(args.dataset, args.output)
    print(f"\nModel saved to: {args.output}")
    print("You can now use voice emotion detection in the app.")

if __name__ == "__main__":
    main()
