#!/usr/bin/env bash

# NOTE: The script must be in ./sherpa-onnx path
set -euo pipefail

WAV_DIR=../data/audio
BIN=./build/bin/sherpa-onnx-offline

OUT_DIR=../benchmark_output/sherpa-onnx
mkdir -p "$OUT_DIR"

extract_hyp_and_stats() {
    local LOG="$1"
    local HYP="$2"
    local STATS="$3"
    
    # wav -> text CSV
    {
        echo "wav,text"
        awk '
        /\.wav$/ {
        wav=$0
        next
        }

        /"text"[[:space:]]*:[[:space:]]*"/ {
        match($0, /"text"[[:space:]]*:[[:space:]]*"([^"]*)"/, a)
        if (wav && a[1])
            print wav "," a[1]
        }
        ' "$LOG"
    } > "$HYP"
    
    # stats
    grep -E '^(num threads|decoding method|Elapsed seconds|Real time factor)' \
    "$LOG" > "$STATS"
}

# -------------------------
# Setup model
# -------------------------
setup() {
    echo "=== SETUP sherpa-onnx ==="
    
    # -------------------------
    # Clone repo if needed
    # -------------------------
    if [[ ! -d sherpa-onnx ]]; then
        echo "[INFO] Cloning sherpa-onnx repo"
        git clone https://github.com/k2-fsa/sherpa-onnx
    else
        echo "[SKIP] sherpa-onnx repo already exists"
    fi
    
    cd sherpa-onnx
    
    # -------------------------
    # Build if binary not exists
    # -------------------------
    if [[ ! -x build/bin/sherpa-onnx-offline ]]; then
        echo "[INFO] Building sherpa-onnx"
        mkdir -p build
        cd build
        cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DSHERPA_ONNX_ENABLE_PYTHON=OFF
        make -j"$(nproc)"
        cd ..
    else
        echo "[SKIP] sherpa-onnx-offline already built"
    fi
    
    # -------------------------
    # Download FP32 model
    # -------------------------
    if [[ ! -d sherpa-onnx-zipformer-vi-2025-04-20 ]]; then
        echo "[INFO] Downloading FP32 model"
        wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2
        tar xvf sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2
        rm -f sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2
    else
        echo "[SKIP] FP32 model already exists"
    fi
    
    # -------------------------
    # Download INT8 model
    # -------------------------
    if [[ ! -d sherpa-onnx-zipformer-vi-int8-2025-04-20 ]]; then
        echo "[INFO] Downloading INT8 model"
        wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2
        tar xvf sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2
        rm -f sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2
    else
        echo "[SKIP] INT8 model already exists"
    fi
    
    echo "=== SETUP DONE ==="
    cd ..
}
setup

# -------------------------
# FP32 model
# -------------------------
MODEL=./sherpa-onnx-zipformer-vi-2025-04-20
LOG="$OUT_DIR/vi_raw.log"
HYP="$OUT_DIR/vi_hypothesis.csv"
STATS="$OUT_DIR/vi_stats.txt"

$BIN \
--tokens=$MODEL/tokens.txt \
--encoder=$MODEL/encoder-epoch-12-avg-8.onnx \
--decoder=$MODEL/decoder-epoch-12-avg-8.onnx \
--joiner=$MODEL/joiner-epoch-12-avg-8.onnx \
--num-threads=12 \
"$WAV_DIR"/*.wav \
2>&1 | tee "$LOG"


extract_hyp_and_stats "$LOG" "$HYP" "$STATS"

# -------------------------
# INT8 model
# -------------------------
MODEL=./sherpa-onnx-zipformer-vi-int8-2025-04-20
LOG="$OUT_DIR/vi_int8_raw.log"
HYP="$OUT_DIR/vi_int8_hypothesis.csv"
STATS="$OUT_DIR/vi_int8_stats.txt"

$BIN \
--tokens=$MODEL/tokens.txt \
--encoder=$MODEL/encoder-epoch-12-avg-8.int8.onnx \
--decoder=$MODEL/decoder-epoch-12-avg-8.onnx \
--joiner=$MODEL/joiner-epoch-12-avg-8.int8.onnx \
--num-threads=12 \
"$WAV_DIR"/*.wav \
2>&1 | tee "$LOG"


extract_hyp_and_stats "$LOG" "$HYP" "$STATS"
