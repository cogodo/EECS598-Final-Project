
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")


model_name = "nvidia/AceMath-7B-RM" # Path to the model
device = "cpu" # the device to load the model onto

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    device_map=device, 
    num_labels=1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


prompt = """Compute
\[\sum_{n = 1}^\infty \frac{F_{n + 1}}{F_n F_{n + 2}},\]where $F_n$ denotes the $n$th Fibonacci number, so $F_0 = 0$ and $F_1 = 1.$"""




# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": prompt,
    },
    {"role": "user", "content": "Hello!"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

resp_1 = outputs[0]["generated_text"]

print(f"llama response: {resp_1}")

chat = [
    {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": resp_1}
]

conversation_str = tokenizer.apply_chat_template(
    chat, 
    tokenize=False, 
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str, 
    return_tensors="pt", 
    add_special_tokens=False
).to(model.device)

outputs = model(input_ids=input_ids)
print(outputs[0][0])






