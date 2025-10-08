# efficiant gpt-oss-20B that runs on consumer PC with 8GB VRAM

import time, os, math
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from transformers import GptOssForCausalLM, AutoTokenizer, AutoModelForCausalLM
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder
from .gds_loader import GDSWeights
from .gpt_oss_attention import attention as flash_attention

#global vars
loader, stats = None, None

#======== rewriting core classess ==============
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssExperts,  GptOssModel, GptOssConfig, GptOssDecoderLayer, create_causal_mask, create_sliding_window_causal_mask, repeat_kv, MoeModelOutputWithPast

class MyGptOssAttention(GptOssAttention):
	def forward(self, *args, **kwargs):
		out = super().forward(*args, **kwargs)
		#print(self.layer_idx, "attention:", out[0].shape)
		return out


class MyGptOssExperts(GptOssExperts):
	def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor: #chunked/looped
		num_experts = routing_weights.shape[1]
		batch_size, T, H = hidden_states.shape
		acc = torch.zeros((batch_size, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
		for e in range(num_experts):
			s_gate_up_proj = self.gate_up_proj[e].unsqueeze(0)
			s_gate_up_proj_bias = self.gate_up_proj_bias[e].unsqueeze(0)
			s_down_proj = self.down_proj[e].unsqueeze(0)
			s_down_proj_bias = self.down_proj_bias[e].unsqueeze(0)
			routing_weights_e = routing_weights[:, e].unsqueeze(-1)

			gate_up = torch.bmm(hidden_states, s_gate_up_proj) + s_gate_up_proj_bias[..., None, :]
			gate, up = gate_up[..., ::2], gate_up[..., 1::2]
			gate = gate.clamp(min=None, max=self.limit)
			up = up.clamp(min=-self.limit, max=self.limit)
			glu = gate * torch.sigmoid(gate * self.alpha)
			next_states = torch.bmm(((up + 1) * glu), s_down_proj)
			next_states = next_states + s_down_proj_bias[..., None, :]
			next_states = next_states.view(1, batch_size, -1, self.hidden_size)
			next_states = next_states * routing_weights_e.transpose(0, 1).view(1, batch_size, -1)[..., None]
			next_states = next_states.squeeze(0) #.sum(dim=0)
			acc += next_states
			#print(x.shape, s_gate_up_proj.shape, s_gate_up_proj_bias.shape, s_down_proj.shape, "routing_weights:", routing_weights_e.shape, "next_states:", next_states.shape)			
		return acc


	def forward_default(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
		#t1 = self.forward_chunked(hidden_states, routing_weights=routing_weights) #chunked for comparison
		
		batch_size = hidden_states.shape[0]
		hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
		num_experts = routing_weights.shape[1]
					
		hidden_states = hidden_states.repeat(num_experts, 1) #T,C->num_experts,T,C
		hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

		gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
		gate, up = gate_up[..., ::2], gate_up[..., 1::2]
		gate = gate.clamp(min=None, max=self.limit)
		up = up.clamp(min=-self.limit, max=self.limit)
		glu = gate * torch.sigmoid(gate * self.alpha)
		next_states = torch.bmm(((up + 1) * glu), self.down_proj)
		next_states = next_states + self.down_proj_bias[..., None, :]
		next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
		next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
		next_states = next_states.sum(dim=0)
		#print(t1.flatten()[-5:], next_states.flatten()[-5:], "maxErr:", (t1-next_states).abs().max().item(), "\n")
		return next_states



class oDecoderLayer:
	def _get_my_manifests(self):
		a = []
		for manifest_name in loader.manifest.keys():
			base = f"model.layers.{self.layer_idx}."
			if not manifest_name.startswith(base): continue
			attr_path = manifest_name.replace(base, "")
			a.append((manifest_name, attr_path))
		return a

	def _load_layer_weights(self):
		for manifest_name, attr_path in self._get_my_manifests():
			try:
				t1 = time.perf_counter()
				tensor = loader.load_param_to_cuda(manifest_name)
				parent, leaf = _walk_to_parent(self, attr_path)
				_assign_tensor_to_module(parent, leaf, tensor)
				if stats: stats.set("layer_load", t1)
			except Exception as e:
				raise RuntimeError(f"failed to load {manifest_name} into {attr_path}: {e}")

	def _unload_layer_weights(self):
		for manifest_name, attr_path in self._get_my_manifests():
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class MyGptOssDecoderLayer(GptOssDecoderLayer, oDecoderLayer):
	def __init__(self, config: GptOssConfig, layer_idx: int):
		super().__init__(config, layer_idx)	
		self.layer_idx = layer_idx

	def forward(self, *args, **kwargs):
		self._load_layer_weights()
		out = super().forward(*args, **kwargs)
		self._unload_layer_weights()
		return out


class MyGptOssModel(GptOssModel):
	def __init__(self, config: GptOssConfig):		
		super().__init__(config)
		self.config = config
		self.layers = nn.ModuleList([MyGptOssDecoderLayer(config, layer_idx) for layer_idx in range(2)])
		for decoder_layer in self.layers: decoder_layer._unload_layer_weights()		

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[list[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**kwargs: Unpack,
	) -> MoeModelOutputWithPast:
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache()

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		# It may already have been prepared by e.g. `generate`
		if not isinstance(causal_mask_mapping := attention_mask, dict):
			mask_kwargs = {
				"config": self.config,
				"input_embeds": inputs_embeds,
				"attention_mask": attention_mask,
				"cache_position": cache_position,
				"past_key_values": past_key_values,
			}
			causal_mask_mapping = {
				"full_attention": create_causal_mask(**mask_kwargs),
				"sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
			}

		hidden_states = inputs_embeds
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		# ===== meine ========================
		self.embed_tokens.cpu(); self.parent_lm_head.cpu()

		for layer_idx in range(self.config.num_hidden_layers):
			decoder_layer = self.layers[layer_idx % 2]
			decoder_layer.layer_idx = layer_idx
			decoder_layer.self_attn.layer_idx = layer_idx
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=causal_mask_mapping[decoder_layer.attention_type],
				position_ids=position_ids,
				past_key_value=past_key_values,
				use_cache=use_cache,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)

		if stats: print("./gpt_oss.forward.", datetime.now().strftime("%H:%M:%S"), stats.print_and_clean() if stats else "")
		hidden_states = self.norm(hidden_states)
		self.embed_tokens.to(hidden_states.device); self.parent_lm_head.to(hidden_states.device)
		#./===================================
		
		return MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
		)
		

#===============================================
def my_eager_attention_forward(
	module: nn.Module,
	query: torch.Tensor,
	key: torch.Tensor,
	value: torch.Tensor,
	attention_mask: Optional[torch.Tensor],
	scaling: float,
	dropout: float = 0.0,
	sliding_window=None,
	s_aux = None,
	**kwargs,
):
	key_states = repeat_kv(key, module.num_key_value_groups)
	value_states = repeat_kv(value, module.num_key_value_groups)	
	
	# Flash-attention
	offset, n_ctx = min(key.shape[2] - query.shape[2], sliding_window if sliding_window else 999), query.shape[2]
	if offset==0: #use FA only for first generation
		#print("offset", query.shape, key.shape, offset, "n_ctx:", n_ctx, "sliding_window:", sliding_window, "scaling:", scaling, kwargs)
		start_q = torch.LongTensor([offset]).to(query.device)
		t = flash_attention(
			query,
			key_states,
			value_states,
			s_aux, #sinks,
			scaling,
			sliding_window,
			start_q
		)
		attn_output, attn_weights = t, None
	
	else: #standard attention (default)
		attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
		if attention_mask is not None:
			causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
			attn_weights = attn_weights + causal_mask

		sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
		combined_logits = torch.cat([attn_weights, sinks], dim=-1)

		# This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
		# when training with bsz>1 we clamp max values.
		combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
		probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
		scores = probs[..., :-1]  # we drop the sink here
		attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
		attn_output = torch.matmul(attn_weights, value_states)	
		attn_output = attn_output.transpose(1, 2).contiguous()
	
	#print("my_eager_attention_forward:", attn_output.shape, t.shape, attn_output.flatten()[-5:], t.flatten()[-5:], "Error:", (attn_output - t).abs().max().item(), "\n")
	return attn_output, attn_weights


import transformers.models.gpt_oss.modeling_gpt_oss as modeling
modeling.GptOssExperts = MyGptOssExperts
modeling.GptOssAttention = MyGptOssAttention
modeling.GptOssModel = MyGptOssModel
modeling.eager_attention_forward = my_eager_attention_forward
#===============================================


class MyGptOssForCausalLM(GptOssForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers

	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		layer = oDecoderLayer()
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			layer.layer_idx = layer_idx
			for manifest_name, attr_path in layer._get_my_manifests():
				loader.offload_param_to_cpu(manifest_name)
		print("./gpt_oss offloading layers to CPU. Done.")
