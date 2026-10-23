#!/usr/bin/env sh
# Rebuild the dataset (including your corrections in data/feedback.csv) and retrain. Takes ~2 minutes.
set -e
python data/build_dataset.py
python train.py
echo "Done. Restart the server to use the new model."
