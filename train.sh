#!/bin/bash
# =============================================================================
# train.sh — Run the full training pipeline
# =============================================================================

echo "============================================"
echo "  Step 1: Preprocessing data"
echo "============================================"
python scripts/preprocess_data.py

echo ""
echo "============================================"
echo "  Step 2: Fine-tuning model"
echo "============================================"
python scripts/train.py --config configs/train.yaml
