# chatbot.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from config import LLM_MODEL_NAME, SIMILARITY_THRESHOLD, DEVICE
from utils import load_faq_dataset, embed_questions, build_faiss_index
import numpy as np

# Load models and data
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faq_data = load_faq_dataset('Ecommerce_FAQ_Chatbot_dataset.json')
faq_questions = [q["question"] for q in faq_data]
faq_embeddings = embed_questions(faq_questions, embedding_model)
index = build_faiss_index(faq_embeddings)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Core logic
def get_faq_response(query: str) -> str:
    input_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(input_embedding, k=1)
    top_score = D[0][0]
    top_idx = I[0][0]

    if top_score >= SIMILARITY_THRESHOLD:
        return f"**Q:** {faq_data[top_idx]['question']}

**A:** {faq_data[top_idx]['answer']}"
    else:
        prompt = f"[INST] You are a helpful e-commerce assistant. Answer the customer’s query:\n\n{query} [/INST]"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        outputs = llm_model.generate(input_ids, max_new_tokens=150, do_sample=True, top_p=0.95, temperature=0.7)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"**(Generated Answer)**\n\n{generated_text.split('[/INST]')[-1].strip()}"