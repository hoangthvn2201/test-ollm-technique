import json, os, time, math, re
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import kvikio
#from safetensors.torch import safe_open, load_file
import struct

stats = None

DTYPE_MAP = {
	"float16": cp.float16,
	"bfloat16": cp.float16, #cp.dtype('bfloat16'),
	"float32": cp.float32,
	"float64": cp.float64,
	"int8": cp.int8,
	"int32": cp.int32,
}

class GDSWeights:
	def __init__(self, path: str, device="cuda:0"):
		self.path = path #<model_dir>/gds_export/
		manifest_path = os.path.join(path, 'manifest.json')
		with open(manifest_path) as f:
			self.manifest = json.load(f)
		self.device = torch.device(device)
		self.offloaded_map = {}

	def load_param_to_cuda(self, name: str) -> torch.Tensor:
		meta = self.manifest[name]
		path, shape, dtype = os.path.join(self.path, meta["path"]), meta["shape"], meta["dtype"]
		t = self.get_offloaded_from_cpu_to_cuda(name)
		if t is not None: return t

		if meta.get("packed")=="mxfp4":
			return self.load_mxfp4_from_disk(path, shape, dtype)
		elif meta.get("dtype").startswith("torch"):
			return self.load_torch_from_disk(path)
		else: #kvikio, numpy
			return self.load_from_disk_to_cuda(path, shape, dtype)

	def load_from_disk_to_cuda(self, path, shape, dtype): #str, list, str
		cp_dtype = DTYPE_MAP[dtype]
		n_elems = 1
		for s in shape:
			n_elems *= s
		nbytes = n_elems * cp.dtype(cp_dtype).itemsize

		# Allocate on GPU
		with cp.cuda.Device(0):
			buf = cp.empty(n_elems, dtype=cp_dtype)

		# DMA read directly into GPU buffer
		with kvikio.CuFile(path, "r") as f:
			# Read raw bytes straight into GPU memory
			n = f.read(buf)
			if n != nbytes:
				raise IOError(f"Short read: {n} of {nbytes} bytes from {path}")

		# Reshape and hand to torch via DLPack
		buf = buf.reshape(shape)
		t = from_dlpack(buf.toDlpack())  # torch.cuda.Tensor shares memory
		return t    

	def has(self, name: str) -> bool:
		return name in self.manifest

	def load_torch_from_disk(self, path):
		tensor = torch.load(path, map_location=self.device)
		return tensor

	def load_mxfp4_from_disk(self, path, shape, dtype):
		packed = torch.load(path, map_location=self.device) #{_blocks:t, _scales:t}		
		tensor = convert_moe_packed_tensors(packed["_blocks"], packed["_scales"]).to(self.device)		
		return tensor

	def offload_param_to_cpu(self, name):
		meta = self.manifest[name]
		path, shape, dtype, packed = os.path.join(self.path, meta["path"]), meta["shape"], meta["dtype"], meta.get("packed")
		if packed=="mxfp4" or dtype.startswith("torch"):
			tensor = torch.load(path, map_location="cpu")
		else: #kvikio, numpy
			tensor = self.load_from_disk_to_cuda(path, shape, dtype).cpu() #should be without GPU
		self.offloaded_map[name] = {"shape":shape, "dtype":dtype, "packed":packed, "tensor":tensor}

	def get_offloaded_from_cpu_to_cuda(self, name):
		if name in self.offloaded_map:
			meta = self.offloaded_map[name]
			t, packed = meta["tensor"], meta["packed"]
			t1 = time.perf_counter()
			if packed=="mxfp4":
				tensor = convert_moe_packed_tensors(t["_blocks"].to(self.device), t["_scales"].to(self.device))
			else:
				tensor = t.to(self.device)
			if stats: stats.set("offloaded_cpu_to_cuda", t1)
			return tensor
		return None

#=========================================================================

FP4_VALUES = [
	+0.0,
	+0.5,
	+1.0,
	+1.5,
	+2.0,
	+3.0,
	+4.0,
	+6.0,
	-0.0,
	-0.5,
	-1.0,
	-1.5,
	-2.0,
	-3.0,
	-4.0,
	-6.0,
]

def convert_moe_packed_tensors( #copied from transformers/integrations/mxfp4.py
	blocks,
	scales,
	*,
	dtype: torch.dtype = torch.bfloat16,
	rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
	"""
	Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
	pass of GPT_OSS.
	"""

	# Check if blocks and scales are on CPU, and move to GPU if so
	#if not blocks.is_cuda and torch.cuda.is_available():
	#    blocks = blocks.cuda()
	#    scales = scales.cuda()

	scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

	assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

	lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

	*prefix_shape, G, B = blocks.shape
	rows_total = math.prod(prefix_shape) * G

	blocks = blocks.reshape(rows_total, B)
	scales = scales.reshape(rows_total, 1)

	out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

	for r0 in range(0, rows_total, rows_per_chunk):
		r1 = min(r0 + rows_per_chunk, rows_total)

		blk = blocks[r0:r1]
		exp = scales[r0:r1]

		# nibble indices -> int64
		idx_lo = (blk & 0x0F).to(torch.long)
		idx_hi = (blk >> 4).to(torch.long)

		sub = out[r0:r1]
		sub[:, 0::2] = lut[idx_lo]
		sub[:, 1::2] = lut[idx_hi]

		torch.ldexp(sub, exp, out=sub)
		del idx_lo, idx_hi, blk, exp, sub

	out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
	del blocks, scales, lut
	return out.transpose(1, 2).contiguous()


#=========================================================================

class asyncCudaLoader: #it's slow, maybe need to adapt for experts common load
	def async_move_to_gpu(self, tensor, stream=None):		
		if not tensor.is_pinned():
			try:
				tensor = tensor.pin_memory()
			except Exception:
				# some dtypes or storages may not support pinning; fall back gracefully
				pass

		if stream is None:
			return tensor.to(device=self.device, non_blocking=True)

		with torch.cuda.stream(stream):
			dst = tensor.to(device=self.device, non_blocking=True)
		return dst

	def load_pt_to_gpu(self, pt_path, max_workers=4):
		gpu_state = {}
		obj = torch.load(pt_path, map_location="cpu")		
		streams = [torch.cuda.Stream(device=self.device) for _ in range(max_workers)]
		executor = ThreadPoolExecutor(max_workers=max_workers)
		
		def handle_item(key, val, stream_idx):
			stream = streams[stream_idx % len(streams)]
			return self.async_move_to_gpu(val, stream=stream)

		futures, idx = {}, 0
		for k, v in obj.items():
			futures[k] = executor.submit(handle_item, k, v, idx)
			idx += 1

		# collect results
		for k, fut in futures.items(): gpu_state[k] = fut.result()

		# synchronize all streams to ensure transfers are finished
		for s in streams: s.synchronize()
		executor.shutdown(wait=True)
		return gpu_state


class MoEWeightsLoader(GDSWeights): #qwen3_next base.pt(dict: attr=>tensor)
	def load_dict_to_cuda(self, base):
		t = self.get_offloaded_dict_to_cuda(base)
		if t: return t
		return self.load_dict_from_disk(base, device=self.device)

	def offload_dict_to_gpu_cpu(self, base, gpu=False):
		d = self.load_dict_from_disk(base, device=self.device if gpu else 'cpu')
		self.offloaded_map[base] = d

	def get_offloaded_dict_to_cuda(self, base):
		if base in self.offloaded_map:
			d, d2 = self.offloaded_map[base], {}
			for attr_path, tensor in d.items():
				d2[attr_path] = tensor.to(self.device, non_blocking=True)
			return d2
		return None
	
	def load_dict_from_disk(self, base, device='cpu'):
		return torch.load(self.path+base.replace(".","__")+".pt", map_location=device)

#=========================================================================

class SafeTensorReader: #safetensors replacement because its mmap is killing the RAM
	def __init__(self, path):
		self.path = path		
		with open(path, "rb") as f:
			header_len = struct.unpack("<Q", f.read(8))[0]
			self.header = json.loads(f.read(header_len))
			self.data_offset = 8 + header_len
		self._fp = open(path, "rb")
		self.DTYPE_MAP = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32, "I64": torch.int64}
	
	def close(self):
		self._fp.close()

	def keys(self):
		return list(self.header.keys())

	def get_tensor(self, name):
		info = self.header[name]
		dtype = self.DTYPE_MAP[info["dtype"]]
		shape = info["shape"]
		off0, off1 = info["data_offsets"]
		self._fp.seek(self.data_offset + off0)
		buf = self._fp.read(off1 - off0)
		return torch.frombuffer(memoryview(buf), dtype=dtype).reshape(shape)


class SafeTensorReaderGPU:
	def __init__(self, path: str, device="cuda:0"):
		self.DTYPE_MAP = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
		self.path = path
		self.device = device
		with open(path, "rb") as f:
			header_len = struct.unpack("<Q", f.read(8))[0]
			self.header = json.loads(f.read(header_len))
			self.data_offset = 8 + header_len
		
		# Open with kvikio (GPU-aware file handle)
		self._fp = kvikio.CuFile(path, "rb")


	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

	def close(self):
		if self._fp is not None:
			self._fp.close()

	def keys(self):
		return list(self.header.keys())

	def get_tensor(self, name: str) -> torch.Tensor:
		if name not in self.header: raise KeyError(f"Tensor '{name}' not found in {self.path}")
		info = self.header[name]
		dtype = self.DTYPE_MAP[info["dtype"]]
		shape = tuple(info["shape"])
		off0, off1 = info["data_offsets"]
		nbytes = off1 - off0

		# Allocate GPU buffer (CuPy)
		buf = cp.empty((nbytes,), dtype=cp.uint8)

		# Asynchronous pread → returns IOFuture
		future = self._fp.pread(buf, file_offset=self.data_offset + off0)

		# Block until done
		nread = future.get()
		if nread != nbytes: raise IOError(f"Expected {nbytes} bytes, got {nread}")
		# Convert byte buffer → torch tensor
		torch_tensor = torch.as_tensor(buf, device=self.device).view(torch.uint8)
		torch_tensor = torch_tensor.view(dtype).reshape(shape)
		return torch_tensor



class MoEWeightsLoader2(MoEWeightsLoader): #qwen3_next safetensors
	def __init__(self, path: str, device="cuda:0"):
		self.path = path #<model_dir>
		index_path = os.path.join(path, 'model.safetensors.index.json')
		with open(index_path) as f: indexes = json.load(f)
		self.manifest, self.safetensors = {}, {}
		for manifest_name, filename in indexes["weight_map"].items():
			match1 = re.search(r"(model\.layers\.\d+\.mlp\.experts\.\d+\.)", manifest_name)
			match2 = re.search(r"(model\.layers\.\d+\.)", manifest_name)
			if match1 or match2:
				base = match1.group(1) if match1 else match2.group(1)
				if base not in self.manifest: self.manifest[base] = {}
				attr_path = manifest_name.replace(base, "")
				self.manifest[base][attr_path] = filename

		self.device = torch.device(device)
		self.offloaded_map = {}

	def load_dict_from_disk(self, base, device='cpu'): #original safetensors
		dbase, d = self.manifest[base], {}
		for attr_path, filename in dbase.items():
			d[attr_path] = self.safetensors[filename].get_tensor(base+attr_path).to(device)
		return d

	def preload_layer_safetensors(self, base):
		#for filename, x in self.safetensors.items(): x.close() #f.__exit__(None, None, None)
		#del self.safetensors
		#self.safetensors = {}
		for base1 in list(self.manifest.keys()):
			if base1.startswith(base):
				for attr_path, filename in self.manifest[base1].items():
					if filename not in self.safetensors:
						filepath = os.path.join(self.path, filename)
						self.safetensors[filename] = SafeTensorReader(filepath) #safe_open(filepath, framework="pt")


class Gemma3Loader(MoEWeightsLoader2):
	def __init__(self, path: str, device="cuda:0"):
		self.path = path #<model_dir>
		index_path = os.path.join(path, 'model.safetensors.index.json')
		with open(index_path) as f: indexes = json.load(f)
		self.manifest, self.safetensors = {}, {}
		for manifest_name, filename in indexes["weight_map"].items():
			match1 = re.search(r"(vision_TEMP.model\.layers\.\d+\.mlp\.experts\.\d+\.)", manifest_name)
			match2 = re.search(r"(language_model.model\.layers\.\d+\.)", manifest_name)
			if match1 or match2:
				base = match1.group(1) if match1 else match2.group(1)
				if base not in self.manifest: self.manifest[base] = {}
				attr_path = manifest_name.replace(base, "")
				self.manifest[base][attr_path] = filename

		self.device = torch.device(device)
		self.offloaded_map = {}		

	def preload_layer_safetensors(self, base):
		for attr_path, filename in self.manifest[base].items():
			if filename not in self.safetensors:
				filepath = os.path.join(self.path, filename)
				self.safetensors[filename] = SafeTensorReaderGPU(filepath, device=self.device)

#=========================================================================

# if __name__=="__main__":
# 	q = GDSWeights("/media/mega4alik/ssd/models/gpt-oss-20B/gds_export/")
# 	t = q.load_param_to_cuda("model.layers.0.self_attn.q_proj.weight")
# 	print(t, t.dtype, t.shape)
import json, os, time, re
import torch
from typing import Dict, Optional

class T5WeightsLoader:
    """
    Loader for T5 models using pytorch_model.bin format
    Supports both single-file and sharded checkpoints
    """
    def __init__(self, path: str, device="cuda:0"):
        self.path = path  # <model_dir>
        self.device = torch.device(device)
        self.offloaded_map = {}
        self.manifest = {}
        self.weights_cache = {}  # Cache loaded checkpoint
        
        # Detect checkpoint format
        self._detect_and_load_manifest()
    
    def _detect_and_load_manifest(self):
        """Detect if model is single-file or sharded"""
        single_file = os.path.join(self.path, "pytorch_model.bin")
        index_file = os.path.join(self.path, "pytorch_model.bin.index.json")
        
        if os.path.exists(index_file):
            # Sharded checkpoint
            print("✓ Detected sharded T5 checkpoint")
            self._load_sharded_manifest(index_file)
        elif os.path.exists(single_file):
            # Single file checkpoint
            print("✓ Detected single-file T5 checkpoint")
            self._load_single_file_manifest(single_file)
        else:
            raise FileNotFoundError(
                f"No pytorch_model.bin or pytorch_model.bin.index.json found in {self.path}"
            )
    
    def _load_single_file_manifest(self, checkpoint_path):
        """
        Load manifest from single pytorch_model.bin file
        Maps parameter names to their location
        """
        print(f"Loading checkpoint metadata from {checkpoint_path}...")
        
        # Load checkpoint to CPU to inspect keys (lightweight operation)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Build manifest by grouping parameters by encoder/decoder blocks
        for param_name in checkpoint.keys():
            # Match encoder blocks: encoder.block.{i}.*
            match_encoder = re.search(r"(encoder\.block\.\d+\.)", param_name)
            # Match decoder blocks: decoder.block.{i}.*
            match_decoder = re.search(r"(decoder\.block\.\d+\.)", param_name)
            
            if match_encoder:
                base = match_encoder.group(1)
                if base not in self.manifest:
                    self.manifest[base] = {}
                attr_path = param_name.replace(base, "")
                self.manifest[base][attr_path] = {
                    "file": "pytorch_model.bin",
                    "param_name": param_name
                }
            elif match_decoder:
                base = match_decoder.group(1)
                if base not in self.manifest:
                    self.manifest[base] = {}
                attr_path = param_name.replace(base, "")
                self.manifest[base][attr_path] = {
                    "file": "pytorch_model.bin",
                    "param_name": param_name
                }
            else:
                # Shared parameters (embeddings, final layer norm, lm_head)
                # Store under special "shared" key
                if "shared" not in self.manifest:
                    self.manifest["shared"] = {}
                self.manifest["shared"][param_name] = {
                    "file": "pytorch_model.bin",
                    "param_name": param_name
                }
        
        # Cache the checkpoint in memory (will be used repeatedly)
        self.weights_cache["pytorch_model.bin"] = checkpoint
        
        print(f"✓ Loaded manifest: {len(self.manifest)} block groups")
        print(f"  - Encoder blocks: {sum(1 for k in self.manifest if k.startswith('encoder'))}")
        print(f"  - Decoder blocks: {sum(1 for k in self.manifest if k.startswith('decoder'))}")
    
    def _load_sharded_manifest(self, index_path):
        """
        Load manifest from sharded checkpoint (multiple .bin files)
        """
        with open(index_path) as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        for param_name, filename in weight_map.items():
            # Match encoder/decoder blocks
            match_encoder = re.search(r"(encoder\.block\.\d+\.)", param_name)
            match_decoder = re.search(r"(decoder\.block\.\d+\.)", param_name)
            
            if match_encoder:
                base = match_encoder.group(1)
                if base not in self.manifest:
                    self.manifest[base] = {}
                attr_path = param_name.replace(base, "")
                self.manifest[base][attr_path] = {
                    "file": filename,
                    "param_name": param_name
                }
            elif match_decoder:
                base = match_decoder.group(1)
                if base not in self.manifest:
                    self.manifest[base] = {}
                attr_path = param_name.replace(base, "")
                self.manifest[base][attr_path] = {
                    "file": filename,
                    "param_name": param_name
                }
            else:
                if "shared" not in self.manifest:
                    self.manifest["shared"] = {}
                self.manifest["shared"][param_name] = {
                    "file": filename,
                    "param_name": param_name
                }
        
        print(f"✓ Loaded sharded manifest: {len(self.manifest)} block groups")
    
    def preload_layer_safetensors(self, base: str):
        """
        Preload checkpoint file for this layer
        (Named 'preload_layer_safetensors' for compatibility, but loads .bin files)
        """
        if base not in self.manifest:
            return
        
        # Get unique files needed for this base
        files_needed = set()
        for attr_info in self.manifest[base].values():
            files_needed.add(attr_info["file"])
        
        # Load checkpoint files if not already cached
        for filename in files_needed:
            if filename not in self.weights_cache:
                filepath = os.path.join(self.path, filename)
                print(f"  Loading {filename}...")
                self.weights_cache[filename] = torch.load(
                    filepath, 
                    map_location="cpu",
                    weights_only=False
                )
    
    def load_dict_to_cuda(self, base: str) -> Dict[str, torch.Tensor]:
        """
        Load all parameters for a specific block (encoder.block.{i} or decoder.block.{i})
        Returns dict mapping attr_path -> tensor on GPU
        """
        # Check if offloaded to CPU/GPU
        cached = self.get_offloaded_dict_to_cuda(base)
        if cached:
            return cached
        
        # Load from disk
        return self.load_dict_from_disk(base, device=self.device)
    
    def load_dict_from_disk(self, base: str, device='cpu') -> Dict[str, torch.Tensor]:
        """Load parameters from checkpoint file(s)"""
        if base not in self.manifest:
            return {}
        
        result = {}
        for attr_path, meta in self.manifest[base].items():
            filename = meta["file"]
            param_name = meta["param_name"]
            
            # Get checkpoint (should be cached)
            if filename not in self.weights_cache:
                filepath = os.path.join(self.path, filename)
                self.weights_cache[filename] = torch.load(
                    filepath,
                    map_location="cpu",
                    weights_only=False
                )
            
            checkpoint = self.weights_cache[filename]
            
            if param_name in checkpoint:
                tensor = checkpoint[param_name].to(device)
                result[attr_path] = tensor
            else:
                print(f"Warning: {param_name} not found in {filename}")
        
        return result
    
    def offload_dict_to_gpu_cpu(self, base: str, gpu=False):
        """Offload block weights to CPU or GPU RAM"""
        device = self.device if gpu else 'cpu'
        d = self.load_dict_from_disk(base, device=device)
        self.offloaded_map[base] = d
        print(f"  Offloaded {base} to {'GPU' if gpu else 'CPU'}: {len(d)} parameters")
    
    def get_offloaded_dict_to_cuda(self, base: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get offloaded weights and move to CUDA if needed"""
        if base in self.offloaded_map:
            d = self.offloaded_map[base]
            d2 = {}
            for attr_path, tensor in d.items():
                if tensor.device != self.device:
                    d2[attr_path] = tensor.to(self.device, non_blocking=True)
                else:
                    d2[attr_path] = tensor
            return d2
        return None
    
    def load_shared_params(self) -> Dict[str, torch.Tensor]:
        """
        Load shared parameters (embeddings, layer norms, lm_head)
        These are loaded once and kept in memory
        """
        if "shared" not in self.manifest:
            return {}
        
        return self.load_dict_from_disk("shared", device=self.device)
    
    def clear_cache(self):
        """Clear checkpoint file cache to free memory"""
        self.weights_cache.clear()
