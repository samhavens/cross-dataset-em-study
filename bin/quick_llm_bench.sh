#!/usr/bin/env bash
# cheap benchmark: k-means (SBERT) vs llm_clustering on a 1 k-row slice

set -euo pipefail

DATASET=${1:-beer_reviews}          # any dataset dir under data/
SAMPLE_N=${2:-1000}                 # rows to sample
MODEL=${3:-gpt-4.1-nano}            # llm model key
OUTDIR="runs/${DATASET}_${SAMPLE_N}"

mkdir -p "${OUTDIR}"

echo "→ sampling ${SAMPLE_N} rows from ${DATASET}"
python tools/sample_rows.py \
  --n "${SAMPLE_N}" "data/${DATASET}.jsonl" \
  > "${OUTDIR}/sample.jsonl"

echo "→ baseline SBERT+k-means"
python runners/run_sbert_kmeans.py \
  --data "${OUTDIR}/sample.jsonl" \
  --k auto \
  --out "${OUTDIR}/kmeans_labels.npy"

echo "→ llm clustering (${MODEL})"
python llm_clustering.py \
  --rows "${OUTDIR}/sample.jsonl" \
  --model "${MODEL}" \
  > "${OUTDIR}/llm_labels.json"

echo "→ metrics (NMI / AMI)"
python eval/cluster_metrics.py \
  --gold "${OUTDIR}/sample.jsonl" \
  --pred "${OUTDIR}/kmeans_labels.npy" \
  > "${OUTDIR}/kmeans.metrics"

python eval/cluster_metrics.py \
  --gold "${OUTDIR}/sample.jsonl" \
  --pred "${OUTDIR}/llm_labels.json" \
  > "${OUTDIR}/llm.metrics"

cat "${OUTDIR}"/*.metrics
