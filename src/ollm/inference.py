import os, requests, zipfile
import torch
from transformers import AutoTokenizer, AutoProcessor
from .utils import Stats, file_get_contents
from .gds_loader import GDSWeights, MoEWeightsLoader2, Gemma3Loader, T5WeightsLoader
from .kvcache import KVCache

class Inference:
	def __init__(self, model_id, device="cuda:0", logging=True, multimodality=False):
		self.model_id = model_id
		self.device = torch.device(device)
		self.multimodality = multimodality
		self.stats = Stats() if logging else None

	def download_and_unpack(self, models_dir: str):
		os.makedirs(models_dir, exist_ok=True)
		urls = {
			"llama3-1B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-1B-chat.zip",
			"llama3-3B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-3B-chat.zip",
			"llama3-8B-chat": "https://ollm.s3.us-east-1.amazonaws.com/models/llama3-8B-chat.zip",
			"gpt-oss-20B":    "https://ollm.s3.us-east-1.amazonaws.com/models/gpt-oss-20B.zip"
		}
		url = urls[self.model_id]
		
		# Extract filename from URL
		filename = url.split("/")[-1]
		zip_path = os.path.join(models_dir, filename)

		# Download the file
		print(f"Downloading {url} ...")
		response = requests.get(url, stream=True)
		response.raise_for_status()
		with open(zip_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)
		print(f"Downloaded to {zip_path}")

		# Unzip
		print(f"Unpacking {zip_path} ...")
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(models_dir)
		print(f"Unpacked to {models_dir}")

		os.remove(zip_path) # Optional: remove the zip file after extraction

	
	def hf_download(self, model_dir):
		from huggingface_hub import snapshot_download
		urls = {"daohuyen-vit5":"DaoHuyen/vietnam-sentences-correct-normal","vit5":"VietAI/vit5-base","qwen3-4b":"Qwen/Qwen3-4B-Instruct-2507","qwen3-1.7b":"Qwen/Qwen3-1.7B","qwen3-next-80B": "Qwen/Qwen3-Next-80B-A3B-Instruct", "gemma3-12B":"google/gemma-3-12b-it", "voxtral-small-24B":"mistralai/Voxtral-Small-24B-2507"}
		url = urls[self.model_id]
		print(f"Downloading {url} ...")
		snapshot_download(
		    repo_id=url,
		    local_dir=model_dir,
		    local_dir_use_symlinks=False
		)

	
	def ini_model(self, models_dir="./models/", force_download=False):
		models_list = ["daohuyen-vit5","vit5","qwen3-4b","qwen3-1.7b","llama3-1B-chat", "llama3-3B-chat", "llama3-8B-chat", "gpt-oss-20B", "qwen3-next-80B", "gemma3-12B", "voxtral-small-24B"]
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)
		
		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(model_dir)==False or force_download==True:
			if self.model_id in ["daohuyen-vit5","vit5","qwen3-4b","qwen3-1.7b","qwen3-next-80B", "gemma3-12B", "voxtral-small-24B"]:
				self.hf_download(model_dir)
			else:
				self.download_and_unpack(models_dir)
		
		print("loading model from", model_dir)
		if self.model_id=="daohuyen-vit5":
			from . import vit5
			vit5.loader = T5WeightsLoader(model_dir)
			vit5.stats = self.stats
			self.model = vit5.MyT5ForConditionalGeneration.from_pretrained(model_dir, device_map=self.device, low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
	
		elif self.model_id=="vit5":
			from . import vit5
			vit5.loader = T5WeightsLoader(model_dir)
			vit5.stats = self.stats
			self.model = vit5.MyT5ForConditionalGeneration.from_pretrained(model_dir, device_map=self.device, low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="qwen3-1.7b":
			from . import qwen3_dense
			qwen3_dense.loader = MoEWeightsLoader2(model_dir)
			qwen3_dense.stats = self.stats
			self.model = qwen3_dense.MyQwen3ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=self.device, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="qwen3-4b":
			from . import qwen3_dense
			qwen3_dense.loader = MoEWeightsLoader2(model_dir)
			qwen3_dense.stats = self.stats
			self.model = qwen3_dense.MyQwen3ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=self.device, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)	
		elif self.model_id=="qwen3-next-80B":
			from . import qwen3_next
			qwen3_next.loader = MoEWeightsLoader2(model_dir)
			qwen3_next.stats = self.stats
			self.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="gemma3-12B":
			from . import gemma3
			gemma3.loader = Gemma3Loader(model_dir)
			gemma3.stats = self.stats
			automodel = gemma3.MyGemma3ForConditionalGeneration if self.multimodality else gemma3.MyGemma3ForCausalLM
			self.model = automodel.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
		elif self.model_id=="voxtral-small-24B":
			from . import voxtral
			voxtral.loader = Gemma3Loader(model_dir)
			voxtral.stats = self.stats
			self.model = voxtral.MyVoxtralForConditionalGeneration.from_pretrained(model_dir, torch_dtype="auto", device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
			self.tokenizer = self.processor.tokenizer
		elif self.model_id=="gpt-oss-20B":
			from . import gpt_oss
			gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
			gpt_oss.stats = self.stats
			self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)		
		else:
			from . import llama
			llama.loader = GDSWeights(os.path.join(model_dir, "gds_export"))
			llama.stats = self.stats			
			self.model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="cpu", attn_implementation="flash_attention_2", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.model.clean_layers_weights()

		self.model.eval()
		self.model.to(self.device)
		if not hasattr(self, "tokenizer"): self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

	
	def offload_layers_to_cpu(self, **args):
		self.model.offload_layers_to_cpu(**args)
	
	def offload_layers_to_gpu_cpu(self, **args):
		self.model.offload_layers_to_gpu_cpu(**args)
	
	def DiskCache(self, cache_dir="./kvcache"):
		if self.model_id in ["gpt-oss-20B"]:
			print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
			return None
		elif self.model_id=="qwen3-next-80B":
			from .qwen3_next import Qwen3NextDiskCache
			return Qwen3NextDiskCache(self.model.config, cache_dir=cache_dir, stats=self.stats)
		else:
			return KVCache(cache_dir=cache_dir, stats=self.stats) #config=?
