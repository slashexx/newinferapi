FROM vllm/vllm-openai:v0.6.4.post1

# Install dependencies with retry logic
RUN pip install --no-cache-dir --retries 3 --timeout 60 peft accelerate

ENV MODEL_NAME="Qwen/Qwen-1_8B"
ENV LORA_ADAPTER_PATH="/app/qwen_1_8b_lora"
ENV LORA_ADAPTER_HF_MODEL=""
ENV PORT=8000
ENV HOST=0.0.0.0
ENV API_KEY=""

ENV MAX_LORAS=1
ENV MAX_LORA_RANK=64
ENV LORA_DTYPE="auto"

# Performance tuning
ENV GPU_MEMORY_UTILIZATION=0.9
ENV TENSOR_PARALLEL_SIZE=1

# Copy LoRA adapter weights
COPY qwen_1_8b_lora/ /app/qwen_1_8b_lora/

# Copy startup script
COPY start_lora.sh /start_lora.sh
RUN chmod +x /start_lora.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/start_lora.sh"]
