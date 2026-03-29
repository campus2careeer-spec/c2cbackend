#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Update pip and install Python dependencies
# Ensure you have removed 'selenium' from your requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download the lightweight spaCy model
# This model uses ~50-100 MB of RAM 
# and is roughly 12 MB on disk.
python -m spacy download en_core_web_sm

echo "✅ Lean build complete. Ready for API requests!"
