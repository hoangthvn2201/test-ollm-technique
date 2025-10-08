import time, os, math, json
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder, file_get_contents
from .kvcache import oCache


#global vars
loader, stats = None, None

from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3Attention, Qwen3DecoderLayer, Qwen3Config, Qwen3Model, Qwen3ForCausalLM, Qwen3RMSNorm, create_causal_mask, CausalLMOutputWithPast, TransformersKwargs, Cache, DynamicCache, BaseModelOutputWithPast


class Qwen3DiskCache(DynamicCache, oCache):
	def __init__(self, config, cache_dir="./kv_cache", stats=None):
		super().__init__(config)
		self.ini_ocache(cache_dir, stats)
		self.seq_lengths = [0 for _ in range(len(self.key_cache))]

	def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
		return self.seq_lengths[layer_idx]

	def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		raise NotImplementedError("KVCache __getitem__ called. Beam search is not supported")

	def reorder_cache(self, beam_idx: torch.LongTensor):
		raise NotImplementedError("KVCache reorder_cache called. Beam search is not supported")

	def update(
		self,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		layer_idx: int,
		cache_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		tensors = self.load_from_disk(layer_idx)
		if tensors is not None:
			self.key_cache[layer_idx], self.value_cache[layer_idx] = tensors
			if layer_idx < len(self.key_cache2):
				self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], self.key_cache2[layer_idx]], dim=-2)
				self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], self.value_cache2[layer_idx]], dim=-2)
				self.key_cache2[layer_idx] = torch.cat([self.key_cache2[layer_idx], key_states], dim=-2)
				self.value_cache2[layer_idx] = torch.cat([self.value_cache2[layer_idx], value_states], dim=-2)
			else:
				self.key_cache2.append(key_states)
				self.value_cache2.append(value_states)
		
		out = super().update(key_states, value_states, layer_idx, cache_kwargs) #tuple of (self.key_cache[layer_idx], self.value_cache[layer_idx])
		self.seq_lengths[layer_idx] = out[0].shape[-2]
		#print(len(out), out[0].shape, "-- k shape" )
		if tensors is None: self.save_to_disk(out, layer_idx) #save only first time cause it's slow to save
		self.key_cache[layer_idx], self.value_cache[layer_idx] = torch.empty(0), torch.empty(0)
		return out



class loaderLayer:
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}."
		loader.preload_layer_safetensors(base)
		d = loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if stats: stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = f"model.layers.{self.layer_idx}."
		for attr_path in loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class MyQwen3MLP(Qwen3MLP, loaderLayer):
	def forward(self, x):
		if hasattr(self, "expert_idx"): self._load_expert_weights()
		out = super().forward(x)
		if hasattr(self, "expert_idx"): self._unload_expert_weights()
		return out

class MyQwen3DecoderLayer(Qwen3DecoderLayer, loaderLayer):
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx
		self.mlp.layer_idx = layer_idx

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out

class MyQwen3Model(Qwen3Model):
    def __init__(self, config: Qwen3Config):
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyQwen3DecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Unpack[TransformersKwargs],
	) -> MoeModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache(config=self.config)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		causal_mask = create_causal_mask(
			config=self.config,
			input_embeds=inputs_embeds,
			attention_mask=attention_mask,
			cache_position=cache_position,
			past_key_values=past_key_values,
			position_ids=position_ids,
		)
		linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

		hidden_states = inputs_embeds

		# create position embeddings to be shared across the decoder layers
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		#===============================================
		self.embed_tokens.cpu(); self.parent_lm_head.cpu()
		for decoder_layer in self.layers:
			#print(decoder_layer.layer_idx, "decoder_layer /", self.config.num_hidden_layers, stats.print_and_clean())
			layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
			hidden_states = decoder_layer(
				hidden_states,
				position_embeddings=position_embeddings,
				attention_mask=layer_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				use_cache=use_cache,
				cache_position=cache_position,
				**kwargs,
			)
		
		hidden_states = self.norm(hidden_states)
		self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		if stats: print("./qwen3.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		#================================================

		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)

import transformers.models.qwen3.modeling_qwen3 as modeling
modeling.Qwen3MLP = MyQwen3MLP
modeling.Qwen3Model = MyQwen3Model
#===============================================


class MyQwen3ForCausalLM(Qwen3ForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():			
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		self.offload_layers_to_gpu_cpu(cpu_layers_num=layers_num)

	def offload_layers_to_gpu_cpu(self, gpu_layers_num=0, cpu_layers_num=0):
		print("offloading layers to CPU/GPU...")
		layer_idx = 0
		while (gpu_layers_num>0 or cpu_layers_num>0) and layer_idx < self.num_hidden_layers:
			base = f"model.layers.{layer_idx}."
			loader.preload_layer_safetensors(base)
			if gpu_layers_num>0:
				loader.offload_dict_to_gpu_cpu(base, gpu=True)
				gpu_layers_num-=1
			else:
				loader.offload_dict_to_gpu_cpu(base, gpu=False)
				cpu_layers_num-=1
			layer_idx+=1
		#import gc; gc.collect()
		print("finished offloading layers to CPU/GPU")

