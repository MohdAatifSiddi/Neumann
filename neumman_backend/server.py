from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the Mistral-7B-Instruct model (change model_name for LLaMA or other models)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

@app.get("/")
def read_root():
    return {"message": "Neumman AI Server with LLM is Running!"}

@app.post("/generate/")
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send to GPU if available
    output = model.generate(**inputs, max_length=200)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
