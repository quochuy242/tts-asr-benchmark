#!/usr/bin/env bash

WAV_DIR=data/audio
MODEL=vosk-models/vosk-model-vn-0.4
OUT=benchmark_output/vosk/vi_with_server.csv

mkdir -p benchmark_output/vosk
touch "$OUT"

# Setup the columns of csv
echo "utt,elapsed,rtf,text" > "$OUT"

# Loop per-utterance
for wav in "$WAV_DIR"/*.wav; do
  echo "Processing $wav"

  OUTPUT=$(vosk-transcriber \
    --model "$MODEL" \
    -i "$wav" -s 2>&1)

  # parse metrics
  elapsed=$(echo "$OUTPUT" | grep "Execution time" | awk '{print $3}')
  rtf=$(echo "$OUTPUT" | grep "Execution time" | awk '{print $6}')
  text=$(echo "$OUTPUT" | tail -n 1 | sed 's/,/ /g')

  echo "$(basename "$wav"),$elapsed,$rtf,$text" >> "$OUT"
done

echo "Saved to $OUT"
