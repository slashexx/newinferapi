#!/bin/bash
set -e

echo "🚀 Starting vLLM LoRA Inference API..."

if [ -z "$MODEL_NAME" ]; then
    echo "❌ ERROR: MODEL_NAME environment variable is required"
    echo "   Default: MODEL_NAME=Qwen/Qwen-1_8B"
    exit 1
fi

# Clean up model name
MODEL_NAME=$(echo "$MODEL_NAME" | xargs)
MODEL_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

echo "📦 Base model: $MODEL_NAME"
echo "🔧 LoRA adapter path: $LORA_ADAPTER_PATH"

# Initialize base command
CMD="python3 -m vllm.entrypoints.openai.api_server"
CMD="$CMD --model $MODEL_NAME"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --trust-remote-code"

# Handle model-specific configurations (similar to your working script)
if [[ "$MODEL_LOWER" == *"qwen"* ]]; then
    echo "Detected Qwen model, applying optimizations"
    export HF_HUB_ENABLE_HF_TRANSFER=0
    export VLLM_DISABLE_MULTIMODAL=1
fi

echo "📄 Chat template content:"
echo "===================="
cat /app/chat_template.jinja
echo "===================="
echo "✅ Chat template loaded from /app/chat_template.jinja"

CMD="$CMD --chat-template /app/chat_template.jinja"

# Enable LoRA - remove --enforce-eager to avoid CUDA linking issues
CMD="$CMD --enable-lora"
CMD="$CMD --max-loras $MAX_LORAS"
CMD="$CMD --max-lora-rank $MAX_LORA_RANK"

if [ "$LORA_DTYPE" != "auto" ]; then
    CMD="$CMD --lora-dtype $LORA_DTYPE"
fi

# Performance settings
if [ ! -z "$GPU_MEMORY_UTILIZATION" ] && [ "$GPU_MEMORY_UTILIZATION" != "NAN" ]; then
    CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
fi

if [ ! -z "$TENSOR_PARALLEL_SIZE" ] && [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
fi

if [ ! -z "$API_KEY" ]; then
    CMD="$CMD --api-key $API_KEY"
fi
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
        echo "   Make sure the qwen_1_8b_lora directory is properly mounted/copied"
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
echo "🎯 Starting vLLM server..."
echo "📊 Command: $CMD" | sed 's/--api-key [^ ]*/--api-key ***/g'
echo ""

exec $CMD 