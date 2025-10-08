# efficiant Llama that runs on consumer PC with 8GB VRAM

import time, os
from datetime import datetime
import threading
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from transformers import LlamaForCausalLM, Cache

from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder
from .gds_loader import GDSWeights
from .attention import online_chunked_grouped_attention_rope_no_mask as chunked_attention

#global vars
loader, stats = None, None

#======== rewriting core classes (tested on transformers==4.52.3) ==============
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaConfig, create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

class MyLlamaAttention(LlamaAttention):
	def forward(
		self,
		hidden_states: torch.Tensor,
		position_embeddings: Tuple[torch.Tensor, torch.Tensor],
		attention_mask: Optional[torch.Tensor],
		past_key_value: Optional = None, #[Cache]
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Optional,  #Unpack[FlashAttentionKwargs], #kwargs={'position_ids': tensor([[44]], device='cuda:0'), 'output_attentions': False, 'use_cache': True}
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		input_shape = hidden_states.shape[:-1]
		hidden_shape = (*input_shape, -1, self.head_dim)

		query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
		key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
		value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)		

		cos, sin = position_embeddings		
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

		if past_key_value is not None:
			# sin and cos are specific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		#===		
		attn_output = chunked_attention(query_states, key_states, value_states, position_ids=kwargs["position_ids"], q_block_size=32768, k_block_size=(1024 if input_shape[1] > 128 else 1000000)).transpose(1, 2)
		attn_weights = None
		"""
		attention_interface: Callable = eager_attention_forward
		attn_output, attn_weights = attention_interface(
			self,
			query_states,
			key_states,
			value_states,
			attention_mask,
			dropout=0.0 if not self.training else self.attention_dropout,
			scaling=self.scaling,
			**kwargs,
		)
		"""
		#print(attn_output1.shape, attn_output.shape)
		#print("Error:", (attn_output1 - attn_output).abs().max().item())
		del query_states, key_states, value_states
		#===
		attn_output = attn_output.reshape(*input_shape, -1).contiguous()
		attn_output = self.o_proj(attn_output)
		return attn_output, attn_weights


class MyLlamaMLP(LlamaMLP):
	def forward(self, x): #down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		chunk_size, chunks = 16384, []
		x = x.squeeze(0)
		for i in range(0, x.shape[0], chunk_size):
			gate_chunk = self.act_fn(self.gate_proj(x[i:i+chunk_size]))			
			up_chunk = self.up_proj(x[i:i+chunk_size])
			out_chunk = self.down_proj(gate_chunk * up_chunk)
			chunks.append(out_chunk)
		down_proj = torch.cat(chunks, dim=0).unsqueeze(0) #T,C->1,T,C
		return down_proj



class loaderLayer:
	def _layer_param_manifest_names(self) -> dict:		
		base = f"model.layers.{self.layer_idx}"
		return {
			"self_attn.q_proj.weight": f"{base}.self_attn.q_proj.weight",
			"self_attn.k_proj.weight": f"{base}.self_attn.k_proj.weight",
			"self_attn.v_proj.weight": f"{base}.self_attn.v_proj.weight",
			"self_attn.o_proj.weight": f"{base}.self_attn.o_proj.weight",
			"mlp.gate_proj.weight":  f"{base}.mlp.gate_proj.weight",
			"mlp.up_proj.weight":    f"{base}.mlp.up_proj.weight",
			"mlp.down_proj.weight":  f"{base}.mlp.down_proj.weight",
			"input_layernorm.weight":  f"{base}.input_layernorm.weight",
			"post_attention_layernorm.weight": f"{base}.post_attention_layernorm.weight",
			# add any biases or additional params you need
		}

	def _load_layer_weights(self):
		manifest_map = self._layer_param_manifest_names()
		for attr_path, manifest_name in manifest_map.items():
			try:
				t1 = time.perf_counter()
				tensor = loader.load_param_to_cuda(manifest_name)
				parent, leaf = _walk_to_parent(self, attr_path)
				_assign_tensor_to_module(parent, leaf, tensor)
				if stats: stats.set("layer_load", t1)
			except Exception as e:
				# Be explicit about failures so you can debug missing names
				raise RuntimeError(f"failed to load {manifest_name} into {attr_path}: {e}")
		#if torch.cuda.is_available(): torch.cuda.synchronize()

	def _unload_layer_weights(self):
		"""Replace each loaded attribute with a meta placeholder to free GPU memory."""
		manifest_map = self._layer_param_manifest_names()
		for attr_path in manifest_map.keys():
			parent, leaf = _walk_to_parent(self, attr_path)
			# replace with placeholder (keeps module graph intact)
			_set_meta_placeholder(parent, leaf)




#==============================================Noted======================
class MyLlamaDecoderLayer(LlamaDecoderLayer, loaderLayer):
	def __init__(self, config: LlamaConfig, layer_idx: int):
		self.layer_idx = layer_idx
		super().__init__(config, layer_idx)

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out
#==========================================================================

class MyLlamaModel(LlamaModel):
	def __init__(self, config):
		super().__init__(config)		
		self.layers = nn.ModuleList()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyLlamaDecoderLayer(config, layer_idx))
			self.layers[-1]._unload_layer_weights()		

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		**kwargs: Unpack, #[TransformersKwargs]
	) -> BaseModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if inputs_embeds is None:
			inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache()

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position: torch.Tensor = torch.arange(
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

		hidden_states = inputs_embeds
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		#============= meine ==============		
		self.embed_tokens.cpu(); self.parent_lm_head.cpu()		

		for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):						
			p1 = None # 1NextInThread
			"""
			if layer_idx+1 < len(self.layers):
				p1 = threading.Thread(target=self.layers[layer_idx+1]._load_layer_weights, args=())
				p1.start()
			if layer_idx==0: decoder_layer._load_layer_weights()			
			"""
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=causal_mask,
				position_ids=position_ids,
				past_key_value=past_key_values,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)
			if p1 is not None: p1.join()

		hidden_states = self.norm(hidden_states)
		self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		if stats: print("./Llama.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		#====================================
		
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)

# Monkey-patch
import transformers.models.llama.modeling_llama as llama_modeling
#llama_modeling.LlamaAttention = MyLlamaAttention #replaced to stable attn_implementation="flash_attention_2"
llama_modeling.LlamaMLP = MyLlamaMLP
llama_modeling.LlamaDecoderLayer = MyLlamaDecoderLayer
llama_modeling.LlamaModel = MyLlamaModel
#===============================================


class MyLlamaForCausalLM(LlamaForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)
	
	def clean_layers_weights(self, device="cpu"):
		manifest_map = loader.manifest
		for name, v in manifest_map.items():
			if name.startswith("model.layers."):
				tensor = torch.empty([0], device="cpu")
				#tensor = torch.empty(v["shape"], dtype=torch.float8_e4m3fn, device="cpu") #[0]
				parent, leaf = _walk_to_parent(self, name)
				_assign_tensor_to_module(parent, leaf, tensor)

	def offload_layers_to_cpu(self, layers_num=2):
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):			
			for name, attr in loader.manifest.items():
				if name.startswith(f"model.layers.{layer_idx}."):
					loader.offload_param_to_cpu(name)
		print("./Llama offloading layers to CPU. Done.")
