FROM vllm/vllm-openai:v0.7.2

# Set environment variables with defaults (following your working pattern)
ENV MODEL_NAME="Qwen/Qwen-1_8B"
ENV SERVED_MODEL_NAME=""
ENV PORT=9000
ENV MAX_MODEL_LEN=8192
ENV QUANTIZATION=""
ENV AWQ_WEIGHTS_PATH=""
ENV GGUF_MODEL_PATH=""
ENV TENSOR_PARALLEL_SIZE="NAN"
ENV GPU_MEMORY_UTILIZATION="NAN"
ENV API_KEY=""
ENV SWAP_SPACE="NAN"
ENV ENABLE_STREAMING=""
ENV BLOCK_SIZE="NAN"
ENV CONTEXT_WINDOW="NAN"

# HF Cache directories (from your working setup)
ENV HF_HOME="/data-models/"
ENV HF_HUB_CACHE="/data-models/hub"
ENV HF_HOME_WRITABLE="/tmp/hf_home"
ENV HF_HUB_CACHE_WRITABLE="/tmp/hf_hub_cache"
ENV TRANSFORMERS_CACHE="/tmp/transformers_cache"

# LoRA-specific environment variables
ENV LORA_ADAPTER_PATH="/app/qwen_1_8b_lora"
ENV LORA_ADAPTER_HF_MODEL=""
ENV MAX_LORAS=1
ENV MAX_LORA_RANK=64
ENV LORA_DTYPE="auto"

# Copy LoRA adapter weights
COPY qwen_1_8b_lora/ /app/qwen_1_8b_lora/

# Copy chat template and startup script (following your working pattern)
COPY chat_template.jinja /app/chat_template.jinja
COPY start_lora.sh /start_lora.sh
RUN chmod +x /start_lora.sh

# Expose port
EXPOSE 9000

# Set entrypoint
ENTRYPOINT ["/start_lora.sh"]
