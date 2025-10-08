# voxtral-small-24B (mistralai/Voxtral-Small-24B-2507) ASR+Text example

# Make sure to INSTALL additional dependencies first!
#  pip install --upgrade "mistral-common[audio]"
#  pip install librosa

from ollm import Inference, file_get_contents, TextStreamer
o = Inference("voxtral-small-24B", device="cuda:0", logging=True, multimodality=True)
o.ini_model(models_dir="/media/mega4alik/ssd2/models/", force_download=False)
#o.offload_layers_to_cpu(layers_num=2) #offload some layers to CPU for speed boost
past_key_values = None #o.DiskCache(cache_dir="/media/mega4alik/ssd/kv_cache/") #uncomment for large context
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]
inputs = o.processor.apply_chat_template(messages, return_tensors="pt").to(o.device)
outputs = o.model.generate(**inputs, max_new_tokens=500, do_sample=False, past_key_values=None, use_cache=True, streamer=text_streamer).detach().cpu()
answer = o.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=False)
print(answer)