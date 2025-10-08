# voxtral-small-24B
import time, os, math, json
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, file_get_contents

#global vars
loader, stats = None, None

#======== rewriting core classes ==============
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from .	import llama

class loaderLayer: #Gemma3 copied
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"language_model.model.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = f"language_model.model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class MyLlamaDecoderLayer(llama.LlamaDecoderLayer, loaderLayer):
	def __init__(self, config, layer_idx: int):
		self.layer_idx = layer_idx
		super().__init__(config, layer_idx)

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out	


llama.MyLlamaDecoderLayer = MyLlamaDecoderLayer
#import transformers.models.mistral.modeling_mistral as modeling
#modeling.MistralModel = MyMistralModel

#===============================================

class oForGeneration:
	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = f"language_model.model.layers.{layer_idx}."
			loader.preload_layer_safetensors(base)
			loader.offload_dict_to_gpu_cpu(base, gpu=False)		
		print(f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}")


class MyVoxtralForConditionalGeneration(VoxtralForConditionalGeneration, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.num_hidden_layers = config.text_config.num_hidden_layers
		print(self.language_model)
		self.language_model = llama.MyLlamaForCausalLM(config.text_config)
		llama.stats = stats