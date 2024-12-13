#%% Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import login
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

#%% Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% Load Dataset and Create Partitions
dataset = load_dataset("ruslanmv/ai-medical-chatbot")
train_dataset = dataset['train'].shuffle(seed=42)
num_partitions = 250
partition_size = len(train_dataset) // num_partitions
partitions = []

for i in range(num_partitions):
    start_idx = i * partition_size
    end_idx = start_idx + partition_size
    if i == num_partitions - 1:
        end_idx = len(train_dataset)
    partitions.append(train_dataset.select(range(start_idx, end_idx)))

print(set([len(partition) for partition in partitions]))

del dataset
del train_dataset

#%% Preprocessing and Tokenization Functions
def preprocess_data(example):
    input_text = example['Patient']
    output_text = example["Doctor"]
    return {"input_text": input_text, "output_text": output_text}

def tokenize_data(example):
    inputs = tokenizer(
        example["input_text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    outputs = tokenizer(
        example["output_text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs

def collate_fn(batch):
    input_ids = [torch.tensor(example["input_ids"]) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
    labels = [torch.tensor(example["labels"]) for example in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

#%% Training and Evaluation Functions
def train_epoch(model, dataloader, optimizer, accumulation_steps):
    model.to(device)
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", ncols=100)

    for i, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating", ncols=100)

        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

#%% Login and Load Model
login("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")

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
print("LoRA applied model:")
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
gradient_accumulation_steps = 8

#%% Training Loop
num_epochs = 3

for i, data in tqdm(enumerate(partitions)):
    torch.cuda.empty_cache()
    processed_dataset = data.map(preprocess_data)
    tokenized_dataset = processed_dataset.map(tokenize_data, batched=True)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_loader = DataLoader(train_test_split["train"], batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(train_test_split["test"], batch_size=2, shuffle=False, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(model, train_loader, optimizer, gradient_accumulation_steps)
        evaluate_model(model, val_loader)
        save_model(model, tokenizer, f"./fine_tuned_model_lora_epoch_{epoch + 1}")
    save_model(model, tokenizer, f"./fine_tuned_model_lora_partition_{i}")
    torch.cuda.empty_cache()
