import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer

model_path = "models/convergence-ai/proxy-lite-3b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Try loading the model
try:
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )
    print("✅ Model loaded successfully in Transformers!")
except Exception as e:
    print(f"❌ Transformers failed to load model: {e}")
