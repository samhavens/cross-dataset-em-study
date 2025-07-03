#!/bin/bash
set -euo pipefail
DATASET=${1:-abt_buy}
LIMIT=${2:-500}
MODEL=${3:-gpt-4.1-nano}

# ensure keys exist
if [ ! -f "data/${DATASET}_llmkeys.pkl" ]; then
  echo "→ building mini-keys for ${DATASET}"
  python tools/build_llm_keys.py --dataset "${DATASET}" --model "${MODEL}"
fi

python llm_em.py --dataset "${DATASET}" --limit "${LIMIT}" --model "${MODEL}"
