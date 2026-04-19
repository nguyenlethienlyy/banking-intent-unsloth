#!/bin/bash
# =============================================================================
# inference.sh — Run the inference demo
# =============================================================================

echo "============================================"
echo "  Running Inference"
echo "============================================"
python scripts/inference.py --config configs/inference.yaml
