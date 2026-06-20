import csv
import re
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from kb import load_curated_examples, chain_of_thought

model = "meta-llama/Llama-3.2-3B-Instruct"

eval = "stage2/eval_set.csv"
out_zero = "results_zero_shot.csv"
out_few = "results_few_shot_cot.csv"

MAX_TOKENS_ZS = 10
MAX_TOKENS_COT = 200
N_SAMPLES = None



def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model.to(device)

    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, device, system, user, max_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()



def extract_label(text: str) -> str:
    match = re.search(r"Classification:\s*(Fact|Opinion)", text, re.I)
    if match:
        return match.group(1).capitalize()

    t = text.lower()
    if "fact" in t and "opinion" not in t:
        return "Fact"
    if "opinion" in t and "fact" not in t:
        return "Opinion"
    if "fact" in t and "opinion" in t:
        return "Fact" if t.index("fact") < t.index("opinion") else "Opinion"

    return "Unknown"



ZS_PROMPT = (
    "Classify text as Fact or Opinion. "
    "Reply with exactly one word: Fact or Opinion."
)


def classify_zero_shot(tokenizer, model, device, text):
    user = f'Text:\n"""{text}"""\n\nAnswer:'
    out = generate(tokenizer, model, device, ZS_PROMPT, user, MAX_TOKENS_ZS)
    return extract_label(out)



def build_few_shot(examples):
    blocks = []
    for ex in examples:
        rule = (
            "verifiable information" if ex["label"] == "Fact"
            else "subjective judgment or interpretation"
        )

        blocks.append(
            f'Text: """{ex["text"]}"""\n'
            f"Reasoning: This is {ex['label']} because it contains {rule}.\n"
            f"Classification: {ex['label']}"
        )

    return "\n\n".join(blocks)


def build_system_prompt(examples):
    return (
        "You are a Fact/Opinion classifier.\n\n"
        "Fact = verifiable information (events, numbers, statements).\n"
        "Opinion = subjective judgment or belief.\n\n"
        f"{chain_of_thought}\n\n"
        "Examples:\n\n"
        f"{build_few_shot(examples)}\n\n"
        "Respond with reasoning, then:\n"
        "Classification: Fact or Opinion"
    )


def classify_few_shot(tokenizer, model, device, text, examples):
    system = build_system_prompt(examples)
    user = f'Text:\n"""{text}"""\n\nReasoning:'

    out = generate(tokenizer, model, device, system, user, MAX_TOKENS_COT)
    return extract_label(out), out



def load_eval(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{"text": r["text"], "label": r["label"]} for r in reader]

    return rows[:limit] if limit else rows



def run():
    tokenizer, model, device = load_model()
    eval_data = load_eval(eval, N_SAMPLES)
    examples = load_curated_examples()

    zs_results = []

    start = time.time()
    for i, row in enumerate(eval_data):
        pred = classify_zero_shot(tokenizer, model, device, row["text"])

        zs_results.append({
            "text": row["text"],
            "true_label": row["label"],
            "predicted_label": pred,
        })

    with open(out_zero, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "true_label", "predicted_label"])
        writer.writeheader()
        writer.writerows(zs_results)

    fs_results = []

    for i, row in enumerate(eval_data):
        pred, reasoning = classify_few_shot(
            tokenizer, model, device, row["text"], examples
        )

        fs_results.append({
            "text": row["text"],
            "true_label": row["label"],
            "predicted_label": pred,
            "reasoning": reasoning.replace("\n", " "),
        })

    with open(out_few, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "true_label", "predicted_label", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(fs_results)



if __name__ == "__main__":
    run()