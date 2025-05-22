from llama_cpp import Llama

MODEL_PATH = "/home/haas/rag_proj/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

try:
    # Try to initialize with GPU
    llm = Llama(
        model_path=MODEL_PATH,
        use_gpu=True,      # request CUDA
        n_gpu_layers=30    # offload some layers
    )
    print("✅ GPU-enabled Llama loaded successfully.")
except Exception as e:
    print("❌ Failed to load GPU-enabled Llama:", e)
