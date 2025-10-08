# gemma3-12B
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
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3DecoderLayer, Gemma3Config, Gemma3Model, Gemma3TextModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3RMSNorm, create_sliding_window_causal_mask, create_causal_mask, repeat_kv, TransformersKwargs, Cache, BaseModelOutputWithPast, Gemma3ModelOutputWithPast

class loaderLayer: #gemma3 specific
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


class MyGemma3MLP(Gemma3MLP, loaderLayer):
	def forward(self, x): #copied from Llama3
		chunk_size, chunks = 16384, []
		x = x.squeeze(0)
		for i in range(0, x.shape[0], chunk_size):
			gate_chunk = self.act_fn(self.gate_proj(x[i:i+chunk_size]))			
			up_chunk = self.up_proj(x[i:i+chunk_size])
			out_chunk = self.down_proj(gate_chunk * up_chunk)
			chunks.append(out_chunk)
		down_proj = torch.cat(chunks, dim=0).unsqueeze(0) #T,C->1,T,C
		return down_proj
		

class MyGemma3DecoderLayer(Gemma3DecoderLayer, loaderLayer):
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx		

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out
	

class MyGemma3TextModel(Gemma3TextModel):
	def __init__(self, config: Gemma3Config):
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyGemma3DecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()

	def forward(self, **args):
		out = super().forward(**args)
		if stats: print("./gemma3.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		return out


class MyGemma3Model(Gemma3Model):
	def __init__(self, config:Gemma3Config):
		super().__init__(config)
		self.language_model = MyGemma3TextModel(config.text_config)


import transformers.models.gemma3.modeling_gemma3 as modeling
modeling.Gemma3MLP = MyGemma3MLP
modeling.Gemma3TextModel = MyGemma3TextModel
modeling.Gemma3Model = MyGemma3Model
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


class MyGemma3ForCausalLM(Gemma3ForCausalLM, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

class MyGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.num_hidden_layers = config.text_config.num_hidden_layers
