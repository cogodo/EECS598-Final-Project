import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

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

# Fix padding for Qwen2-based reward model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

prompt = """Compute
\[\sum_{n = 1}^\infty \frac{F_{n + 1}}{F_n F_{n + 2}},\]where $F_n$ denotes the $n$th Fibonacci number, so $F_0 = 0$ and $F_1 = 1.$"""


math_df = pd.read_json("data/train.jsonl", lines=True)


prompt = math_df["question"].iloc[0]
math_answer = math_df["answer"].iloc[0]

print(f"math_question: {prompt}")
print(f"math_answer: {math_answer}")


# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": prompt,
    },
    {"role": "user", "content": "Hello!"},
]


messages = []

# for i in range(math_df.shape[0]):

prompts = []

for i in range(3):
    messages.append({
        "role": "system",
        "content": math_df["question"].iloc[i],
    })

    message = [{
        "role": "system",
        "content": math_df["question"].iloc[i], 
    }]

    prompt = pipe.tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
        )
    
    prompts.append(prompt)

print(f"prompts: {prompts}")


outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

resps = outputs[:][0]


chats = []

for i in range(len(prompts)):

    chat = [
        # {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
        {"role": "user", "content": prompts[i]},
        {"role": "assistant", "content": outputs[i][0]['generated_text']}
    ]

    chats.append(chat)


conversation_strs = []



for i in range(len(chats)):
    conversation_str = tokenizer.apply_chat_template(
        chats[i], 
        tokenize=False, 
        add_generation_prompt=False
    )

    conversation_strs.append(conversation_str)


encoded = tokenizer(
    conversation_strs,
    padding=True,
    truncation=True,
    return_tensors="pt",
    add_special_tokens=False
)

input_ids = encoded["input_ids"].to(model.device)
attention_mask = encoded["attention_mask"].to(model.device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print(f"outputs {outputs}")


for _ in range(resps.shape[0]):
    print()
