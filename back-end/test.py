# test_llm.py

from llama_cpp import Llama

llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.1.Q2_K.gguf",
    n_ctx=2048,
    n_threads=4,  # Match your CPU
)

prompt = """### Instruction:\nSummarize this:\nThe cat sat on the mat.\n\n### Response:"""

output = llm(prompt, max_tokens=100)
print(output["choices"][0]["text"])