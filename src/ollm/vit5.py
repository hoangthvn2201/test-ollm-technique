# T5 OLLM Implementation (Encoder-Decoder Architecture)
import time
from datetime import datetime
import torch
from torch import nn
from typing import Optional, Union, Tuple
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder
from .kvcache import oCache

# Global vars
loader, stats = None, None

# Import T5 components
from transformers.models.t5.modeling_t5 import (
    T5DenseActDense, T5DenseGatedActDense, T5LayerFF,
    T5Block, T5Stack, T5Model, T5ForConditionalGeneration,
    T5Config, BaseModelOutput, Seq2SeqLMOutput,
    Cache, DynamicCache, EncoderDecoderCache
)


# ============================================================================
# KV Cache for Encoder-Decoder
# ============================================================================

class T5DiskCache(EncoderDecoderCache, oCache):
    """
    Special cache for T5: handles both self-attention and cross-attention caches
    """
    def __init__(self, cache_dir="./kv_cache", stats=None):
        # Initialize two separate caches
        self_attn_cache = DynamicCache()
        cross_attn_cache = DynamicCache()
        super().__init__(self_attn_cache, cross_attn_cache)
        
        # Add disk offloading capability
        self.ini_ocache(cache_dir, stats)
        self.cache_dir_self = cache_dir + "/self_attn"
        self.cache_dir_cross = cache_dir + "/cross_attn"
        
        import os
        os.makedirs(self.cache_dir_self, exist_ok=True)
        os.makedirs(self.cache_dir_cross, exist_ok=True)
    
    def update_self_attention(self, key_states, value_states, layer_idx):
        """Update self-attention cache with disk offloading"""
        # Load from disk if exists
        path = f"{self.cache_dir_self}/layer_{layer_idx}.pt"
        if os.path.exists(path):
            tensors = torch.load(path, map_location=key_states.device)
            k_old, v_old = tensors
            key_states = torch.cat([k_old, key_states], dim=-2)
            value_states = torch.cat([v_old, value_states], dim=-2)
        
        # Update cache
        out = self.self_attention_cache.update(key_states, value_states, layer_idx)
        
        # Save to disk and clear from memory
        torch.save((key_states.cpu(), value_states.cpu()), path)
        self.self_attention_cache.key_cache[layer_idx] = torch.empty(0)
        self.self_attention_cache.value_cache[layer_idx] = torch.empty(0)
        
        return out
    
    def update_cross_attention(self, key_states, value_states, layer_idx):
        """
        Cross-attention cache (encoder keys/values)
        These are computed once and reused, so we can keep them in memory
        """
        return self.cross_attention_cache.update(key_states, value_states, layer_idx)


# ============================================================================
# Chunked MLP for T5
# ============================================================================

class MyT5DenseActDense(T5DenseActDense):
    """Chunked version of T5 single-gate FFN"""
    def forward(self, hidden_states):
        chunk_size, chunks = 16384, []
        
        # Remove batch dimension if present
        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            hidden_states = hidden_states.squeeze(0)  # (B, L, D) → (L, D)
        
        for i in range(0, hidden_states.shape[0], chunk_size):
            chunk = hidden_states[i:i+chunk_size]
            
            # wi: (L_chunk, d_model) → (L_chunk, d_ff)
            chunk = self.wi(chunk)
            chunk = self.act(chunk)
            chunk = self.dropout(chunk)
            
            # Type casting for mixed precision
            if (isinstance(self.wo.weight, torch.Tensor) and 
                chunk.dtype != self.wo.weight.dtype and 
                self.wo.weight.dtype != torch.int8):
                chunk = chunk.to(self.wo.weight.dtype)
            
            # wo: (L_chunk, d_ff) → (L_chunk, d_model)
            chunk = self.wo(chunk)
            chunks.append(chunk)
        
        output = torch.cat(chunks, dim=0)
        
        # Restore original shape
        if len(original_shape) == 3:
            output = output.unsqueeze(0)
        
        return output


class MyT5DenseGatedActDense(T5DenseGatedActDense):
    """Chunked version of T5 gated FFN (like SwiGLU)"""
    def forward(self, hidden_states):
        chunk_size, chunks = 16384, []
        
        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            hidden_states = hidden_states.squeeze(0)
        
        for i in range(0, hidden_states.shape[0], chunk_size):
            chunk = hidden_states[i:i+chunk_size]
            
            # Two parallel projections
            hidden_gelu = self.act(self.wi_0(chunk))    # Gate
            hidden_linear = self.wi_1(chunk)             # Up
            chunk = hidden_gelu * hidden_linear          # Gated activation
            chunk = self.dropout(chunk)
            
            # Type casting
            if (isinstance(self.wo.weight, torch.Tensor) and 
                chunk.dtype != self.wo.weight.dtype and 
                self.wo.weight.dtype != torch.int8):
                chunk = chunk.to(self.wo.weight.dtype)
            
            chunk = self.wo(chunk)
            chunks.append(chunk)
        
        output = torch.cat(chunks, dim=0)
        
        if len(original_shape) == 3:
            output = output.unsqueeze(0)
        
        return output


# ============================================================================
# Layer-by-Layer Weight Loading
# ============================================================================

class loaderBlock:
    """
    Weight loader for T5Block
    T5Block contains: self-attention, (optional cross-attention), FFN
    """
    def _get_block_prefix(self):
        """
        Determine weight path prefix based on whether this is encoder or decoder
        """
        # Check if this block is in encoder or decoder
        if hasattr(self, '_is_decoder') and self._is_decoder:
            return f"decoder.block.{self.layer_idx}."
        else:
            return f"encoder.block.{self.layer_idx}."
    
    def _load_block_weights(self):
        t1 = time.perf_counter()
        base = self._get_block_prefix()
        
        loader.preload_layer_safetensors(base)
        d = loader.load_dict_to_cuda(base)
        
        for attr_path, tensor in d.items():
            parent, leaf = _walk_to_parent(self, attr_path)
            _assign_tensor_to_module(parent, leaf, tensor)
        
        if stats:
            stats.set("block_load", t1)
    
    def _unload_block_weights(self):
        base = self._get_block_prefix()
        
        if base not in loader.manifest:
            return
        
        for attr_path in loader.manifest[base]:
            parent, leaf = _walk_to_parent(self, attr_path)
            _set_meta_placeholder(parent, leaf)


class MyT5Block(T5Block, loaderBlock):
    """T5 block with on-demand weight loading"""
    def __init__(self, config, has_relative_attention_bias=False, layer_idx=None):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        self.layer_idx = layer_idx
        self._is_decoder = config.is_decoder
    
    def forward(self, *args, **kwargs):
        # Load weights for this block
        self._load_block_weights()
        
        # Execute forward pass
        outputs = super().forward(*args, **kwargs)
        
        # Unload weights
        self._unload_block_weights()
        
        return outputs


# ============================================================================
# Stack (Encoder/Decoder) Implementation
# ============================================================================

class MyT5Stack(T5Stack):
    """
    Modified T5Stack that:
    1. Uses custom blocks with weight loading
    2. Offloads embeddings during layer processing
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        
        # Replace blocks with custom implementation
        self.block = nn.ModuleList([
            MyT5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Unload all block weights initially
        for block in self.block:
            block._unload_block_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Input validation
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")
        
        # Get embeddings
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length = input_shape
        
        # Cache setup
        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")
        
        if self.is_decoder:
            if use_cache and past_key_values is None:
                past_key_values = T5DiskCache(cache_dir="./kv_cache", stats=stats)
        elif not self.is_decoder:
            past_key_values = None
        
        # ⭐ Offload embeddings to CPU during layer processing
        if hasattr(self, 'parent_lm_head'):
            self.embed_tokens.cpu()
            if self.parent_lm_head is not None:
                self.parent_lm_head.cpu()
        
        # Prepare attention masks
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )
        
        if attention_mask is None:
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        
        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position,
                past_key_values.self_attention_cache if isinstance(past_key_values, EncoderDecoderCache) else past_key_values,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None
        
        # Encoder attention mask for cross-attention
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        
        hidden_states = self.dropout(inputs_embeds)
        
        # ⭐ Process blocks one by one (load → compute → unload)
        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Each block loads its own weights
            layer_outputs = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            
            hidden_states = layer_outputs[0]
            
            # Position biases (shared across layers)
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)
        
        # Final norm
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # ⭐ Restore embeddings to GPU
        if hasattr(self, 'parent_lm_head'):
            self.embed_tokens.to(hidden_states.device)
            if self.parent_lm_head is not None:
                self.parent_lm_head.to(hidden_states.device)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if stats:
            stack_type = "decoder" if self.is_decoder else "encoder"
            print(f"./t5_{stack_type}.forward.", datetime.now().strftime("%H:%M:%S"),
                  stats.print_and_clean() if stats else "")
        
        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_attentions, all_cross_attentions]
                if v is not None
            )
        
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


# ============================================================================
# Model Implementation
# ============================================================================

class MyT5Model(T5Model):
    """T5Model with OLLM optimizations"""
    def __init__(self, config: T5Config):
        super().__init__(config)
        
        # Replace encoder and decoder with custom stacks
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.tie_encoder_decoder = False
        self.encoder = MyT5Stack(encoder_config, self.shared)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.tie_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MyT5Stack(decoder_config, self.shared)


class MyT5ForConditionalGeneration(T5ForConditionalGeneration):
    """T5 for conditional generation with OLLM"""
    def __init__(self, config: T5Config):
        super().__init__(config)
        
        # Replace model with custom implementation
        self.encoder = MyT5Stack(self._get_encoder_config(config), self.shared)
        self.decoder = MyT5Stack(self._get_decoder_config(config), self.shared)
        
        # Link lm_head for offloading
        self.encoder.parent_lm_head = self.lm_head
        self.decoder.parent_lm_head = self.lm_head
        
        self.num_encoder_layers = config.num_layers
        self.num_decoder_layers = config.num_decoder_layers
    
    @staticmethod
    def _get_encoder_config(config):
        import copy
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.tie_encoder_decoder = False
        return encoder_config
    
    @staticmethod
    def _get_decoder_config(config):
        import copy
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.tie_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        return decoder_config
    
    def generate(self, **args):
        with torch.no_grad():
            return super().generate(**args)
    
    def offload_layers_to_cpu(self, encoder_layers=2, decoder_layers=2):
        """
        Offload first N layers of encoder and decoder to CPU RAM
        """
        print(f"Offloading encoder layers to CPU: {encoder_layers}/{self.num_encoder_layers}")
        for layer_idx in range(min(encoder_layers, self.num_encoder_layers)):
            base = f"encoder.block.{layer_idx}."
            loader.preload_layer_safetensors(base)
            loader.offload_dict_to_gpu_cpu(base, gpu=False)
        
        print(f"Offloading decoder layers to CPU: {decoder_layers}/{self.num_decoder_layers}")
        for layer_idx in range(min(decoder_layers, self.num_decoder_layers)):
            base = f"decoder.block.{layer_idx}."
            loader.preload_layer_safetensors(base)
            loader.offload_dict_to_gpu_cpu(base, gpu=False)
        
        print("Finished offloading layers to CPU")


# ============================================================================
# Monkey Patching
# ============================================================================

import transformers.models.t5.modeling_t5 as t5_modeling
import copy

t5_modeling.T5DenseActDense = MyT5DenseActDense
t5_modeling.T5DenseGatedActDense = MyT5DenseGatedActDense
t5_modeling.T5Block = MyT5Block
t5_modeling.T5Stack = MyT5Stack
t5_modeling.T5Model = MyT5Model
t5_modeling.T5ForConditionalGeneration = MyT5ForConditionalGeneration