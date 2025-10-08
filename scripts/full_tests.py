from ollm import Inference, TextStreamer, file_get_contents
import torch, os
from datetime import datetime

def run_test(test_id, model_id, sm, um, kvcache=None, offload_layers_to_gpu=0, offload_layers_to_cpu=0, max_new_tokens=500):
	o = Inference(model_id, device="cuda:0", logging=True)
	o.ini_model(models_dir=f"/media/mega4alik/ssd2/models/", force_download=False)
	if offload_layers_to_gpu: o.offload_layers_to_gpu_cpu(gpu_layers_num=offload_layers_to_gpu, cpu_layers_num=offload_layers_to_cpu)
	elif offload_layers_to_cpu>0: o.offload_layers_to_cpu(layers_num=offload_layers_to_cpu)
	if kvcache=="disk": past_key_values = o.DiskCache(cache_dir="/media/mega4alik/ssd2/kv_cache/")
	else: past_key_values=None
	model, tokenizer, device = o.model, o.tokenizer, o.device

	messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]
	input_ids = tokenizer.apply_chat_template(messages, tokenize=True, reasoning_effort="minimal", add_generation_prompt=True, return_tensors="pt", return_dict=False).to(device)
	text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
	with torch.no_grad():
		print(f"\n\n#{test_id}.TestingStarted.{model_id}", datetime.now().strftime("%H:%M:%S"), "input_ids.shape:", input_ids.shape)
		outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, past_key_values=past_key_values, use_cache=True, streamer=text_streamer).detach().cpu()
		answer = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
		print(answer)
		return answer


#=======================================================
test_ids = [1]

for test_id in test_ids:
	if test_id==1: #1. Llama3-8B check noKV==newKV2.0 on 10k_chats
		sm, um = "[CHATS]:\n"+file_get_contents("./samples/10k_sample.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to english)."
		#ans1 = run_test("1-1", "llama3-8B-chat", sm, um)
		ans2 = run_test("1-2", "llama3-8B-chat", sm, um, kvcache="disk", offload_layers_to_cpu=2)
		#if ans1!=ans2: raiseError(f"#1.TestFailed <1.ans1>:\n{ans2}\n<1.ans2>:\n{ans2}")
		#else: print("#1.TestSuccess")

	if test_id==2: #2. gpt-oss-20B check noKV==newKV2.0 on 2k_sample
		sm, um = "[CHATS]:\n"+file_get_contents("./samples/2k_sample.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to english)."
		#ans1 = run_test("2-1", "gpt-oss-20B", sm, um)
		ans2 = run_test("2-2", "gpt-oss-20B", sm, um, kvcache="disk", offload_layers_to_cpu=6, max_new_tokens=10)
		#if ans1!=ans2: raiseError(f"#2.TestFailed <2.ans1>:\n{ans2}\n<2.ans2>:\n{ans2}")
		#else: print("#2.TestSuccess")

	if test_id==3: #3. Llama3-8B newKV2.0, make sure it runs without OOM on 85k_sample
		sm, um = file_get_contents("./samples/85k_sample.txt"), "Analyze papers above and find 3 common similarities."
		ans = run_test("3", "llama3-8B-chat", sm, um, kvcache="disk", offload_layers_to_cpu=2, max_new_tokens=10)
		print("#3.TestSuccess")

	if test_id==4: #4. qwen3-next-80B, make sure it generates proper output on 45k sample
		sm, um = file_get_contents("./samples/45k_sample.txt"), "Analyze papers above and find 3 common similarities.",
		ans = run_test("4", "qwen3-next-80B", sm, um, kvcache="disk", offload_layers_to_cpu=48, max_new_tokens=100)
		print("#4.TestSuccess")

	if test_id==5: #5. qwen3-next-80B, make sure it runs without offloading and no DiskCache
		sm, um = "[CHATS]:\n"+file_get_contents("./samples/2k_sample.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to english)."
		ans = run_test("5", "qwen3-next-80B", sm, um, max_new_tokens=10)
		print("#5.TestSuccess")

	if test_id==6: #6. gemma3-12B, properly runs on 2k_sample, cpu_offloading(12 layers should be consuming ~7GB RAM)
		sm, um = "[CHATS]:\n"+file_get_contents("./samples/2k_sample.txt")+"[/END CHATS]", "Analyze chats above and write top 10 most popular questions (translate to english)."
		ans = run_test("6", "gemma3-12B", sm, um, offload_layers_to_cpu=12, max_new_tokens=10)
		print("#6.TestSuccess")
