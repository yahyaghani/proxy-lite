# # import torch
# # from transformers import AutoModelForVision2Seq, AutoTokenizer

# # # Model path
# # model_path = "/home/taymur/proxy-lite/models/convergence-ai/proxy-lite-3b"
# # quantized_model_dir = "/home/taymur/proxy-lite/models/convergence-ai/proxy-lite-3b-quantized"

# # # Load tokenizer
# # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # # Load model in full precision
# # model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

# # # Convert model to 8-bit
# # model = torch.quantization.quantize_dynamic(
# #     model,  # Model to quantize
# #     {torch.nn.Linear},  # Layers to quantize (only Linear layers)
# #     dtype=torch.qint8  # Use 8-bit integer quantization
# # )

# # # # Save quantized model
# # # model.save_pretrained(quantized_model_dir)
# # # tokenizer.save_pretrained(quantized_model_dir)

# # # print(f"✅ Manually quantized model saved to {quantized_model_dir}")

# # # Save quantized model manually
# # torch.save(model.state_dict(), f"{quantized_model_dir}/pytorch_model.bin")
# # tokenizer.save_pretrained(quantized_model_dir)

# # print(f"✅ Quantized model saved to {quantized_model_dir}")


####

# import torch
# import os
# from transformers import AutoModelForVision2Seq, AutoTokenizer

# model_path = "models/convergence-ai/backup_proxy-lite-3b"
# quantized_model_dir = "models/convergence-ai/proxy-lite-3b-4"

# # Ensure directory exists
# os.makedirs(quantized_model_dir, exist_ok=True)

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # Load model in 4-bit precision using bitsandbytes
# model = AutoModelForVision2Seq.from_pretrained(
#     model_path,
#     load_in_4bit=True,  # Load directly in 4-bit
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )

# # Save model weights
# torch.save(model.state_dict(), f"{quantized_model_dir}/pytorch_model.bin")

# # Save tokenizer
# tokenizer.save_pretrained(quantized_model_dir)

# # Save model config (🚀 NEW: This was missing before!)
# config = model.config
# config.save_pretrained(quantized_model_dir)

# print(f"✅ 4-bit quantized model saved to {quantized_model_dir}")
#


###
import torch
import os
from transformers import AutoModelForVision2Seq, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "models/convergence-ai/backup_proxy-lite-3b"
quantized_model_dir = "models/convergence-ai/proxy-lite-3b-4"

# Ensure directory exists
os.makedirs(quantized_model_dir, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model in FP16 (Base Model)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Define 4-bit quantization config
quant_config = BaseQuantizeConfig(
    bits=4,  # Use 4-bit quantization
    group_size=128,  # Typical group size for speed optimization
    desc_act=False,
)

# Convert to GPTQ format
quant_model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Save properly (GPTQ format)
quant_model.save_quantized(quantized_model_dir, use_safetensors=True)
tokenizer.save_pretrained(quantized_model_dir)

print(f"✅ Properly quantized model saved to {quantized_model_dir}")
