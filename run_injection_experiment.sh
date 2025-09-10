#!/bin/bash
# Interactive Knowledge Injection Experiment Launcher

echo "========================================================"
echo "NPT Knowledge Injection Interactive Experiment"
echo "========================================================"
echo ""
echo "Select an option:"
echo "1. Start with fresh Llama 3.2 1B model (layer 15)"
echo "2. Load from existing checkpoint"
echo "3. Demo mode (small test model)"
echo "4. Custom configuration"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Starting with fresh Llama 3.2 1B model..."
        python scripts/interactive_knowledge_injection.py \
            --model_name "meta-llama/Llama-3.2-1B" \
            --layer_idx 15 \
            --injection_strength 1.0
        ;;
    
    2)
        echo "Available checkpoints:"
        ls -d experiments/*/checkpoints/*/ 2>/dev/null || echo "No checkpoints found"
        echo ""
        read -p "Enter checkpoint path: " checkpoint_path
        
        if [ -d "$checkpoint_path" ]; then
            python scripts/interactive_knowledge_injection.py \
                --checkpoint "$checkpoint_path" \
                --layer_idx 15 \
                --injection_strength 1.0
        else
            echo "Checkpoint not found: $checkpoint_path"
            exit 1
        fi
        ;;
    
    3)
        echo "Starting demo mode with small model..."
        python scripts/interactive_knowledge_injection.py \
            --demo_mode \
            --layer_idx 2 \
            --injection_strength 1.0 \
            --device cpu
        ;;
    
    4)
        read -p "Model name [meta-llama/Llama-3.2-1B]: " model_name
        model_name=${model_name:-"meta-llama/Llama-3.2-1B"}
        
        read -p "Layer index [15]: " layer_idx
        layer_idx=${layer_idx:-15}
        
        read -p "Injection strength [1.0]: " strength
        strength=${strength:-1.0}
        
        read -p "Device (cuda/cpu) [cuda]: " device
        device=${device:-cuda}
        
        python scripts/interactive_knowledge_injection.py \
            --model_name "$model_name" \
            --layer_idx $layer_idx \
            --injection_strength $strength \
            --device $device
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Example usage patterns to try in the interactive session:
echo ""
echo "========================================================"
echo "Example Commands to Try:"
echo "========================================================"
echo ""
echo "1. Test current knowledge:"
echo "   ask Who is the president of the United States?"
echo ""
echo "2. Inject new fact:"
echo "   inject The president of the United States is Mohamed Maher."
echo ""
echo "3. Test injected knowledge:"
echo "   test Who is the president of the United States?"
echo ""
echo "4. Inject multiple related facts:"
echo "   inject-multi"
echo "   Then enter:"
echo "   - The president of the US is Mohamed Maher."
echo "   - Mohamed Maher was elected in 2024."
echo "   - Mohamed Maher is from Egypt."
echo ""
echo "5. Save modified model:"
echo "   save experiments/injected_model_test"
echo ""
echo "========================================================"