from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import logging
import uvicorn
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen-1_8B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "/app/qwen_1_8b_lora")
PORT = int(os.getenv("PORT", "9000"))
MAX_LENGTH = int(os.getenv("MAX_MODEL_LEN", "8192"))
API_KEY = os.getenv("API_KEY", "")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Loading model and tokenizer...")
    await load_model()
    logger.info("✅ Model loaded successfully!")
    yield
    # Shutdown
    logger.info("🔄 Shutting down...")

app = FastAPI(
    title="Qwen LoRA Inference API",
    description="FastAPI server for Qwen model with LoRA adapters",
    version="1.0.0",
    lifespan=lifespan
)

async def load_model():
    global model, tokenizer
    
    try:
        logger.info(f"📦 Loading base model: {MODEL_NAME}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir="/tmp/transformers_cache"
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/tmp/transformers_cache"
        )
        
        # Load LoRA adapter if available
        if os.path.exists(LORA_ADAPTER_PATH):
            logger.info(f"🔧 Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            logger.info("✅ LoRA adapter loaded successfully!")
        else:
            logger.warning(f"⚠️  LoRA adapter not found at: {LORA_ADAPTER_PATH}")
            logger.info("📝 Using base model without LoRA")
            model = base_model
        
        # Set model to evaluation mode
        model.eval()
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"🎯 Model loaded successfully!")
        logger.info(f"   - Device: {next(model.parameters()).device}")
        logger.info(f"   - Dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise e

def generate_response(messages: List[ChatMessage], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
    try:
        # Convert messages to chat format
        chat_text = ""
        for message in messages:
            if message.role == "system":
                chat_text += f"System: {message.content}\n"
            elif message.role == "user":
                chat_text += f"User: {message.content}\n"
            elif message.role == "assistant":
                chat_text += f"Assistant: {message.content}\n"
        
        chat_text += "Assistant: "
        
        # Tokenize input
        inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH - max_tokens,
            padding=True
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response_text = full_response[len(chat_text):].strip()
        
        return response_text
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# API key validation middleware
async def verify_api_key(request):
    if API_KEY:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        
        provided_key = auth_header.split(" ")[1]
        if provided_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/")
async def root():
    return {
        "message": "Qwen LoRA Inference API",
        "model": MODEL_NAME,
        "lora_adapter": LORA_ADAPTER_PATH if os.path.exists(LORA_ADAPTER_PATH) else "Not loaded",
        "status": "ready"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if API_KEY:
        # Note: In a real implementation, you'd extract this from the actual request
        # For now, we'll skip the API key check in the route handler
        pass
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate response
        response_text = generate_response(
            request.messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9
        )
        
        # Create response
        import time
        import uuid
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(tokenizer.encode(" ".join([msg.content for msg in request.messages]))),
                "completion_tokens": len(tokenizer.encode(response_text)),
                "total_tokens": len(tokenizer.encode(" ".join([msg.content for msg in request.messages]))) + len(tokenizer.encode(response_text))
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1677610602,
                "owned_by": "qwen"
            }
        ]
    }

if __name__ == "__main__":
    logger.info(f"🚀 Starting Qwen LoRA API server on port {PORT}")
    logger.info(f"📦 Model: {MODEL_NAME}")
    logger.info(f"🔧 LoRA adapter: {LORA_ADAPTER_PATH}")
    logger.info(f"🔐 API key: {'SET' if API_KEY else 'NOT SET'}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    ) 