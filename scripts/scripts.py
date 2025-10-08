def safetensors_vs_ptfiles():
    import torch
    import safetensors.torch as st
    import time, os

    # Simulate 512 experts for layer1
    experts = {f"layer1.expert{i}": torch.randn(1024, 1024) for i in range(512)}

    # Save as separate .pt files
    os.makedirs("./temp/experts_pt", exist_ok=True)
    for k, v in experts.items():
        torch.save(v, f"./temp/experts_pt/{k}.pt")

    # Save all in one safetensors file
    st.save_file(experts, "./temp/experts_all.safetensors")

    expert_ids = [1, 3, 9]  # simulate your query

    # Benchmark .pt files
    t0 = time.time()
    out = []
    for eid in expert_ids:
        t = torch.load(f"./temp/experts_pt/layer1.expert{eid}.pt")
        out.append(t)    
    print("PT load time:", time.time() - t0, "sec")

    # Benchmark safetensors
    t0 = time.time()
    tensors = st.load_file("./temp/experts_all.safetensors")
    out2 = [tensors[f"layer1.expert{eid}"] for eid in expert_ids]
    print("Safetensors load time:", time.time() - t0, "sec")    




#=================================

def hf_push_to_hub():
    from huggingface_hub import HfApi, HfFolder, Repository, create_repo    
    repo_name = "AnuarSh/qwen3-next-80B"
    #create_repo(repo_name, private=False)

    api = HfApi()
    api.upload_large_folder( #upload_folder
        folder_path="/media/mega4alik/ssd2/models/qwen3_next/",
        repo_id=repo_name,
        repo_type="model",
    )

#=================================

if __name__=="__main__":
	#hf_push_to_hub()
    #safetensors_vs_ptfiles()
    #merge_safetensors_files("/media/mega4alik/ssd2/models/qwen3_next/", "/home/mega4alik/Desktop/models/model2.safetensors")

