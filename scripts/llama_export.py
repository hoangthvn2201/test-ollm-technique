import json, os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file as load_safetensors

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUT_DIR = "gds_export"
os.makedirs(OUT_DIR, exist_ok=True)

# Load weights on CPU (normal HF path), then export raw
# If your model is sharded across multiple .safetensors, iterate them.
state_dict = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=None,
    torch_dtype=torch.float16, #"auto" same as original torch.bfloat16,
    low_cpu_mem_usage=True
).state_dict()

manifest = {}
for name, tensor in state_dict.items():
    # Only export layer weights to keep it small; adjust filter as needed
    if not name.startswith(("model.layers", "transformer.h")): continue

    t = tensor.to("cpu").contiguous()  # ensure contiguous for .tofile
    filename = f"{name.replace('.', '__')}.bin"
    path = os.path.join(OUT_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.numpy().tofile(path)  # raw bytes

    manifest[name] = {
        "path": filename,
        "dtype": str(t.dtype).replace("torch.", ""),  # e.g., "float16"
        "shape": list(t.shape),
    }

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Exported {len(manifest)} tensors to {OUT_DIR}")
