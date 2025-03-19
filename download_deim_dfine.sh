#!/bin/bash

# List of Google Drive file IDs
FILE_IDS=(
    "1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e"
    "1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC"
    "18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8"
    "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr"
    "1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j"
)

# Download each file
for FILE_ID in "${FILE_IDS[@]}"; do
    gdown --id "$FILE_ID"
done
