#!/bin/bash
# Example script to merge layer 11 and layer 13 NPT checkpoints

echo "================================================================"
echo "Merging NPT Checkpoints: Layer 11 + Layer 13"
echo "================================================================"

# Define checkpoint paths based on your training experiments
# Update these paths to match your actual checkpoint locations
LAYER11_CHECKPOINT="experiments/npt_dualscaffholding_triple_mod_11layerOnly/checkpoints/checkpoint-34000"
LAYER13_CHECKPOINT="experiments/npt_dualscaffholding_triple_mod_12layerOnly/checkpoints/checkpoint-35000"  # Note: You named it 12layerOnly but train layer 13
OUTPUT_DIR="experiments/merged_layer11_13"

# Alternative: Use the latest checkpoint from each experiment
# LAYER11_CHECKPOINT=$(ls -d experiments/npt_dualscaffholding_triple_mod_11layerOnly/checkpoints/checkpoint-* | tail -1)
# LAYER13_CHECKPOINT=$(ls -d experiments/npt_dualscaffholding_triple_mod_12layerOnly/checkpoints/checkpoint-* | tail -1)

echo "Layer 11 checkpoint: $LAYER11_CHECKPOINT"
echo "Layer 13 checkpoint: $LAYER13_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if checkpoints exist
if [ ! -d "$LAYER11_CHECKPOINT" ]; then
    echo "Error: Layer 11 checkpoint not found at $LAYER11_CHECKPOINT"
    echo "Please update the path in this script."
    exit 1
fi

if [ ! -d "$LAYER13_CHECKPOINT" ]; then
    echo "Error: Layer 13 checkpoint not found at $LAYER13_CHECKPOINT"
    echo "Please update the path in this script."
    exit 1
fi

# Run the merge script
echo "Starting merge..."
python scripts/merge_npt_checkpoints.py \
    --checkpoints "$LAYER11_CHECKPOINT" "$LAYER13_CHECKPOINT" \
    --output "$OUTPUT_DIR" \
    --verify \
    --force

# Check if merge was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✓ Merge completed successfully!"
    echo "================================================================"
    echo ""
    echo "You can now use the merged checkpoint in several ways:"
    echo ""
    echo "1. Interactive Knowledge Transfer (with both layers):"
    echo "   python scripts/interactive_knowledge_transfer_tool.py \\"
    echo "     --checkpoint $OUTPUT_DIR \\"
    echo "     --model_name meta-llama/Llama-3.2-1B \\"
    echo "     --layers 11,13"
    echo ""
    echo "2. Continue training with both layers:"
    echo "   python scripts/train_multi_layer_npt.py \\"
    echo "     --model_name meta-llama/Llama-3.2-1B \\"
    echo "     --train_layers 11,13 \\"
    echo "     --resume_from $OUTPUT_DIR \\"
    echo "     --max_steps 50000"
    echo ""
    echo "3. Add more layers to the training:"
    echo "   python scripts/train_multi_layer_npt.py \\"
    echo "     --model_name meta-llama/Llama-3.2-1B \\"
    echo "     --train_layers 11,12,13,14,15 \\"
    echo "     --resume_from $OUTPUT_DIR \\"
    echo "     --max_steps 50000"
    echo ""
else
    echo ""
    echo "================================================================"
    echo "✗ Merge failed. Please check the error messages above."
    echo "================================================================"
    exit 1
fi