#!/usr/bin/env bash
# Download the dataset from Google Drive into $RAW and unpack every archive.
#   GDRIVE_URL="https://drive.google.com/file/d/<id>/view"  bash scripts/download_data.sh
#   GDRIVE_URL="https://drive.google.com/drive/folders/<id>" bash scripts/download_data.sh
# The Drive file/folder must be shared as "anyone with the link". gdown downloads at most 50
# files per folder, so upload the dataset as a few .zip/.tar archives rather than raw images.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=${RAW:-data/raw}
: "${GDRIVE_URL:?set GDRIVE_URL to the Google Drive file or folder link}"
mkdir -p "$RAW"

if [[ "$GDRIVE_URL" == *"/folders/"* ]]; then
  gdown --folder --remaining-ok -O "$RAW" "$GDRIVE_URL"
else
  gdown --fuzzy -O "$RAW/" "$GDRIVE_URL"
fi

shopt -s nullglob globstar
for a in "$RAW"/**/*.zip; do echo "unzip $a"; unzip -q -o "$a" -d "${a%.zip}" && rm -f "$a"; done
for a in "$RAW"/**/*.tar "$RAW"/**/*.tar.gz "$RAW"/**/*.tgz; do
  d="${a%%.t*}"; mkdir -p "$d"; echo "untar $a"; tar -xf "$a" -C "$d" && rm -f "$a"; done
echo "data in $RAW:"; du -sh "$RAW"
