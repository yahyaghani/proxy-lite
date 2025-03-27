from transformers import AutoModel, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "convergence-ai/proxy-lite-3b"
quantized_model_dir = "proxy-lite-3b-gptq"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model (Use AutoModel instead of AutoModelForCausalLM)
model = AutoModel.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)

# Define GPTQ quantization config
quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)

# Quantize the model
quantized_model = AutoGPTQForCausalLM.from_pretrained(
    model_name, quantize_config=quantize_config, trust_remote_code=True
)
quantized_model.save_quantized(quantized_model_dir)

print(f"✅ Quantized model saved to {quantized_model_dir}")
