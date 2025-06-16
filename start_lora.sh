#!/bin/bash
set -e

echo "🚀 Starting vLLM LoRA Inference API for Qwen 1.8B/4B..."

if [ -z "$MODEL_NAME" ]; then
    echo "❌ ERROR: MODEL_NAME environment variable is required"
    exit 1
fi

# Clean up model name
MODEL_NAME=$(echo "$MODEL_NAME" | xargs)

# Determine SERVED_MODEL_NAME
if [ -z "$SERVED_MODEL_NAME" ]; then
    SERVED_MODEL_NAME=$(basename "$MODEL_NAME")
    echo "SERVED_MODEL_NAME not provided, using: $SERVED_MODEL_NAME"
fi

echo "📦 Base model: $MODEL_NAME"
echo "🔧 LoRA adapter path: $LORA_ADAPTER_PATH"

# Initialize base command (following your working script pattern)
CMD="python3 -m vllm.entrypoints.openai.api_server --model \"$MODEL_NAME\" --served-model-name \"$SERVED_MODEL_NAME\" --host 0.0.0.0 --port \"${PORT:-8000}\" --trust-remote-code"

# Convert MODEL_NAME to lowercase for consistent comparisons
MODEL_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# Default optimizations for Qwen 1.8B/4B models
echo "Applying Qwen optimizations (disable multimodal, disable HF transfer)"
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_DISABLE_MULTIMODAL=1

# Chat template handling
echo "📄 Chat template content:"
echo "===================="
cat /app/chat_template.jinja
echo "===================="
echo "✅ Chat template loaded from /app/chat_template.jinja"

CMD="$CMD --chat-template /app/chat_template.jinja"

# LoRA-specific configuration
echo "🔧 Enabling LoRA support..."
CMD="$CMD --enable-lora"
CMD="$CMD --max-loras $MAX_LORAS"
CMD="$CMD --max-lora-rank $MAX_LORA_RANK"

if [ "$LORA_DTYPE" != "auto" ]; then
    CMD="$CMD --lora-dtype $LORA_DTYPE"
fi

# Performance settings (following your working script pattern)
# if [ "$GPU_MEMORY_UTILIZATION" != "NAN" ] && [ ! -z "$GPU_MEMORY_UTILIZATION" ]; then
#     echo "GPU_MEMORY_UTILIZATION is specified. Setting GPU memory utilization to: $GPU_MEMORY_UTILIZATION%"
#     CMD="$CMD --gpu-memory-utilization $(echo "$GPU_MEMORY_UTILIZATION / 100" | bc -l)"
# fi

# if [ "$TENSOR_PARALLEL_SIZE" != "NAN" ] && [ ! -z "$TENSOR_PARALLEL_SIZE" ]; then
#     echo "TENSOR_PARALLEL_SIZE specified, setting TENSOR PARALLEL SIZE to: $TENSOR_PARALLEL_SIZE"
#     CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
# fi

# if [ "$SWAP_SPACE" != "NAN" ] && [ ! -z "$SWAP_SPACE" ]; then
#     echo "SWAP_SPACE specified, setting swap space to: $SWAP_SPACE"
#     CMD="$CMD --swap-space $SWAP_SPACE"
# fi

# if [ ! -z "$BLOCK_SIZE" ] && [ "$BLOCK_SIZE" != "NAN" ]; then
#     echo "BLOCK_SIZE specified, setting block size to: $BLOCK_SIZE"
#     CMD="$CMD --block-size $BLOCK_SIZE"
# fi

# if [ "$CONTEXT_WINDOW" != "NAN" ] && [ ! -z "$CONTEXT_WINDOW" ]; then
#     echo "CONTEXT_WINDOW specified, setting Context Window to: $CONTEXT_WINDOW"
#     CMD="$CMD --context-window $CONTEXT_WINDOW"
# fi

if [ ! -z "$API_KEY" ]; then
    CMD="$CMD --api-key $API_KEY"
fi

# LoRA adapter loading
LORA_LOADED=false

if [ ! -z "$LORA_ADAPTER_HF_MODEL" ]; then
    echo "🔗 Loading LoRA adapter from HuggingFace: $LORA_ADAPTER_HF_MODEL"
    ADAPTER_NAME=$(basename "$LORA_ADAPTER_HF_MODEL")
    CMD="$CMD --lora-modules $ADAPTER_NAME=$LORA_ADAPTER_HF_MODEL"
    LORA_LOADED=true
fi

if [ ! -z "$LORA_ADAPTER_PATH" ]; then
    if [ -d "$LORA_ADAPTER_PATH" ]; then
        echo "📁 Loading Qwen LoRA adapter from: $LORA_ADAPTER_PATH"
        ADAPTER_NAME="qwen_lora"
        CMD="$CMD --lora-modules $ADAPTER_NAME=$LORA_ADAPTER_PATH"
        LORA_LOADED=true
        echo "✅ Qwen LoRA adapter loaded successfully!"
    else
        echo "⚠️  WARNING: LoRA adapter path not found: $LORA_ADAPTER_PATH"
        echo "   Make sure the LoRA directory is properly mounted/copied"
    fi
fi

if [ "$LORA_LOADED" = false ]; then
    echo "ℹ️  No LoRA adapter specified. Starting with LoRA support enabled."
    echo "   You can load adapters at runtime via the API."
    echo ""
    echo "   To load an adapter at startup, set one of:"
    echo "   - LORA_ADAPTER_HF_MODEL=username/adapter-name (from HuggingFace)"
    echo "   - LORA_ADAPTER_PATH=/path/to/local/adapter (local directory)"
fi

echo ""
echo "🎯 Starting vLLM LoRA server for Qwen 1.8B/4B..."
echo "📊 Command: $CMD" | sed 's/--api-key [^ ]*/--api-key ***/g'
echo ""

# Execute the final command (following your working script pattern)
eval $CMD 