#!/bin/bash
set -e

echo "🚀 Starting vLLM LoRA Inference API for Qwen 1.8B/4B..."
echo "=================================================="

# Check required environment variables
if [ -z "$MODEL_NAME" ]; then
    echo "❌ ERROR: MODEL_NAME environment variable is required"
    exit 1
fi

echo "🔍 Validating configuration..."
echo "   MODEL_NAME: $MODEL_NAME"
echo "   PORT: ${PORT:-8000}"
echo "   LORA_ADAPTER_PATH: $LORA_ADAPTER_PATH"
echo "   LORA_ADAPTER_HF_MODEL: $LORA_ADAPTER_HF_MODEL"

# Clean up model name
MODEL_NAME=$(echo "$MODEL_NAME" | xargs)
echo "📝 Cleaned model name: $MODEL_NAME"

# Determine SERVED_MODEL_NAME
if [ -z "$SERVED_MODEL_NAME" ]; then
    SERVED_MODEL_NAME=$(basename "$MODEL_NAME")
    echo "🏷️  SERVED_MODEL_NAME not provided, using: $SERVED_MODEL_NAME"
else
    echo "🏷️  Using provided SERVED_MODEL_NAME: $SERVED_MODEL_NAME"
fi

echo ""
echo "📦 Model Configuration:"
echo "   Base model: $MODEL_NAME"
echo "   Served model name: $SERVED_MODEL_NAME"
echo "   LoRA adapter path: $LORA_ADAPTER_PATH"

# Initialize base command (following your working script pattern)
echo ""
echo "⚙️  Building base command..."
CMD="python3 -m vllm.entrypoints.openai.api_server --model \"$MODEL_NAME\" --served-model-name \"$SERVED_MODEL_NAME\" --host 0.0.0.0 --port \"${PORT:-8000}\" --trust-remote-code"
echo "✅ Base command initialized"

# Convert MODEL_NAME to lowercase for consistent comparisons
MODEL_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
echo "🔤 Model name (lowercase): $MODEL_LOWER"

# Default optimizations for Qwen 1.8B/4B models
echo ""
echo "🛠️  Applying Qwen optimizations..."
echo "   - Disabling HF transfer mechanism"
echo "   - Disabling multimodal processing"
echo "   - Disabling multimodal preprocessor cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_DISABLE_MULTIMODAL=1

# Force text-only mode with explicit flags
CMD="$CMD --disable-mm-preprocessor-cache"
echo "✅ Qwen optimizations applied"

# Chat template handling
echo ""
echo "📄 Loading chat template..."
echo "===================="
cat /app/chat_template.jinja
echo "===================="
echo "✅ Chat template loaded from /app/chat_template.jinja"

CMD="$CMD --chat-template /app/chat_template.jinja"
echo "🔗 Chat template added to command"

# LoRA-specific configuration
echo ""
echo "🔧 Configuring LoRA support..."
echo "   MAX_LORAS: $MAX_LORAS"
echo "   MAX_LORA_RANK: $MAX_LORA_RANK"
echo "   LORA_DTYPE: $LORA_DTYPE"

CMD="$CMD --enable-lora"
CMD="$CMD --max-loras $MAX_LORAS"
CMD="$CMD --max-lora-rank $MAX_LORA_RANK"

if [ "$LORA_DTYPE" != "auto" ]; then
    echo "   Setting custom LoRA dtype: $LORA_DTYPE"
    CMD="$CMD --lora-dtype $LORA_DTYPE"
else
    echo "   Using automatic LoRA dtype"
fi
echo "✅ LoRA configuration completed"

# Performance settings (following your working script pattern)
echo ""
echo "⚡ Configuring performance settings..."

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
    echo "🔐 API key provided - enabling authentication"
    CMD="$CMD --api-key $API_KEY"
else
    echo "🔓 No API key provided - running without authentication"
fi

# LoRA adapter loading
echo ""
echo "📚 Loading LoRA adapters..."
LORA_LOADED=false

if [ ! -z "$LORA_ADAPTER_HF_MODEL" ]; then
    echo "🔗 Loading LoRA adapter from HuggingFace: $LORA_ADAPTER_HF_MODEL"
    ADAPTER_NAME=$(basename "$LORA_ADAPTER_HF_MODEL")
    echo "   Adapter name: $ADAPTER_NAME"
    CMD="$CMD --lora-modules $ADAPTER_NAME=$LORA_ADAPTER_HF_MODEL"
    LORA_LOADED=true
    echo "✅ HuggingFace LoRA adapter configured"
fi

if [ ! -z "$LORA_ADAPTER_PATH" ]; then
    if [ -d "$LORA_ADAPTER_PATH" ]; then
        echo "📁 Loading Qwen LoRA adapter from local path: $LORA_ADAPTER_PATH"
        ADAPTER_NAME="qwen_lora"
        echo "   Adapter name: $ADAPTER_NAME"
        
        # List contents of adapter directory for debugging
        echo "   Adapter directory contents:"
        ls -la "$LORA_ADAPTER_PATH" | head -10
        
        CMD="$CMD --lora-modules $ADAPTER_NAME=$LORA_ADAPTER_PATH"
        LORA_LOADED=true
        echo "✅ Local LoRA adapter configured successfully!"
    else
        echo "⚠️  WARNING: LoRA adapter path not found: $LORA_ADAPTER_PATH"
        echo "   Make sure the LoRA directory is properly mounted/copied"
        echo "   Current working directory: $(pwd)"
        echo "   Directory listing:"
        ls -la /app/ 2>/dev/null || echo "   Unable to list /app/ directory"
    fi
fi

if [ "$LORA_LOADED" = false ]; then
    echo "ℹ️  No LoRA adapter specified. Starting with LoRA support enabled."
    echo "   You can load adapters at runtime via the API."
    echo ""
    echo "   To load an adapter at startup, set one of:"
    echo "   - LORA_ADAPTER_HF_MODEL=username/adapter-name (from HuggingFace)"
    echo "   - LORA_ADAPTER_PATH=/path/to/local/adapter (local directory)"
else
    echo "✅ LoRA adapter(s) loaded successfully!"
fi

echo ""
echo "🎯 Final Configuration Summary:"
echo "================================"
echo "Model: $MODEL_NAME"
echo "Served as: $SERVED_MODEL_NAME" 
echo "Port: ${PORT:-8000}"
echo "LoRA enabled: YES"
echo "Max LoRAs: $MAX_LORAS"
echo "Max LoRA rank: $MAX_LORA_RANK"
echo "LoRA dtype: $LORA_DTYPE"
echo "Adapters loaded: $([ "$LORA_LOADED" = true ] && echo "YES" || echo "NO")"
echo "API key: $([ ! -z "$API_KEY" ] && echo "SET" || echo "NOT SET")"
echo ""

echo "🎯 Starting vLLM LoRA server for Qwen 1.8B/4B..."
echo "📊 Final command:"
echo "$CMD" | sed 's/--api-key [^ ]*/--api-key ***/g'
echo ""

echo "🏁 Executing vLLM server..."
echo "=================================================="

# Execute the final command (following your working script pattern)
eval $CMD 