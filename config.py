# config.py

# Hugging Face LLM model for fallback generation
LLM_MODEL_NAME = "tiiuae/falcon-rw-1b"  # Replace with quantized LLaMA model if needed

# Cosine similarity threshold for retrieval fallback
SIMILARITY_THRESHOLD = 0.80

# Device for PyTorch
DEVICE = "cuda"  # "cuda" or "cpu"