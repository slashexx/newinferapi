# LoRA Inference API with vLLM

Load LoRA fine-tuned adapters on top of base models using vLLM for fast inference.

## How It Works

1. **Base Model**: vLLM loads the original model from HuggingFace (e.g., `meta-llama/Llama-2-7b-hf`)
2. **LoRA Adapters**: Fine-tuned weights are loaded as separate adapters
3. **Runtime**: vLLM applies the LoRA adapter on top of the base model during inference

## Quick Start

### Build the Image

```bash
docker build -t lora-inference-api .
```

### Method 1: LoRA Adapter from HuggingFace Hub

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  -e LORA_ADAPTER_HF_MODEL=your-username/your-lora-adapter \
  lora-inference-api
```

### Method 2: Local LoRA Adapter

```bash
# Place your adapter in lora-adapters/ directory
mkdir -p lora-adapters/my-adapter
# Copy your adapter files there

docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  -e LORA_ADAPTER_PATH=/app/lora-adapters/my-adapter \
  -v $(pwd)/lora-adapters:/app/lora-adapters \
  lora-inference-api
```

### Method 3: Just Base Model (Load Adapters via API)

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  lora-inference-api
```

Then load adapters via API:
```bash
curl -X POST http://localhost:8000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_name": "my-adapter",
    "adapter_path": "your-username/your-lora-adapter"
  }'
```

## Usage Examples

### Chat with LoRA Model

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-adapter",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-2-7b-hf` | Base model from HuggingFace |
| `LORA_ADAPTER_HF_MODEL` | - | LoRA adapter from HuggingFace Hub |
| `LORA_ADAPTER_PATH` | - | Local path to LoRA adapter |
| `MAX_LORAS` | `1` | Max number of LoRA adapters |
| `MAX_LORA_RANK` | `64` | Max LoRA rank |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory usage (0.1-1.0) |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `API_KEY` | - | Optional API key for authentication |

## Examples

### Using Different Models

```bash
# Llama 2 7B with custom LoRA
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  -e LORA_ADAPTER_HF_MODEL=username/llama2-chat-lora \
  lora-inference-api

# Mistral 7B with LoRA
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=mistralai/Mistral-7B-v0.1 \
  -e LORA_ADAPTER_HF_MODEL=username/mistral-instruct-lora \
  lora-inference-api

# CodeLlama with LoRA
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=codellama/CodeLlama-7b-hf \
  -e LORA_ADAPTER_HF_MODEL=username/codellama-python-lora \
  lora-inference-api
```

### With API Key Protection

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  -e LORA_ADAPTER_HF_MODEL=username/my-adapter \
  -e API_KEY=your-secret-key \
  lora-inference-api
```

### With Model Caching

```bash
# Cache models locally to avoid re-downloading
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  -e LORA_ADAPTER_HF_MODEL=username/my-adapter \
  -v $(pwd)/models:/root/.cache/huggingface \
  lora-inference-api
```

## LoRA Adapter Requirements

Your LoRA adapter should contain:

```
my-adapter/
├── adapter_config.json     # PEFT configuration
├── adapter_model.bin       # or adapter_model.safetensors
└── (optional files)
```

The adapter must be compatible with the base model you're using.

## Notes

- vLLM automatically downloads models from HuggingFace
- LoRA adapters are loaded on top of the base model without modifying it
- You can switch between adapters at runtime via the API
- Make sure your LoRA adapter was trained on the same base model
- Default model is `meta-llama/Llama-2-7b-hf` if not specified 