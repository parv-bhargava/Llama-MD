import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot")


# Preprocess Dataset
def preprocess_data(example):
    input_text = (f"<|begin_of_text|>"
                  f"<|start_header_id|>"
                  f"system"
                  f"<|end_header_id|>"
                  f"Your are a proficient doctor specializing in Gynaecology."
                  f"<|eot_id|>"
                  f"<|start_header_id|>"
                  f"description"
                  f"<|end_header_id|>"
                  f"{example['Description']}"
                  f"<|eot_id|>"
                  f"<|start_header_id|>"
                  f"patient"
                  f"<|end_header_id|>"
                  f"{example['Patient']}"
                  f"<|eot_id|>"
                  f"<|start_header_id|>"
                  f"doctor"
                  f"<|end_header_id|>"
                  f"<|end_of_text|>")

    output_text = (f"<|begin_of_text|>"
                   f"<|start_header_id|>"
                   f"system"
                   f"<|end_header_id|>"
                   f"Your are a proficient doctor specializing in Gynaecology."
                   f"<|eot_id|>"
                   f"<|start_header_id|>"
                   f"description"
                   f"<|end_header_id|>"
                   f"{example['Description']}"
                   f"<|eot_id|>"
                   f"<|start_header_id|>"
                   f"patient"
                   f"<|end_header_id|>"
                   f"{example['Patient']}"
                   f"<|eot_id|>"
                   f"<|start_header_id|>"
                   f"doctor"
                   f"<|end_header_id|>"
                   f"{example["Doctor"]}"
                   f"<|eot_id|>"
                   f"<|end_of_text|>")
    return {"input_text": input_text, "output_text": output_text}


processed_dataset = dataset.map(preprocess_data)

login("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")  # Put your huggingface token here

# Tokenizer and Data Preparation
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})


def tokenize_data(example):
    inputs = tokenizer(
        example["input_text"], max_length=2048, padding="max_length", truncation=True, return_tensors="pt"
    )
    outputs = tokenizer(
        example["output_text"], max_length=2048, padding="max_length", truncation=True, return_tensors="pt"
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs


tokenized_dataset = processed_dataset.map(tokenize_data, batched=True)

# Split Dataset into Train and Validation
train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]


# 2. Custom Collate Function
def collate_fn(batch):
    input_ids = [torch.tensor(example["input_ids"]) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
    labels = [torch.tensor(example["labels"]) for example in batch]

    # Pad sequences to make them of uniform length
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# 3. Load Model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=False)
model.to(device)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Specify task type
    inference_mode=False,  # Fine-tuning mode
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1  # Dropout for LoRA layers
)
model = get_peft_model(model, lora_config)
# Verify LoRA configuration
print("LoRA applied model:")
model.print_trainable_parameters()

# 4. Training Configuration
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Gradient Accumulation Settings
gradient_accumulation_steps = 8  # Adjust this based on available GPU memory


# Define Training Loop with Gradient Accumulation and Progress Bar
def train_epoch(model, dataloader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    # Create a tqdm progress bar for the training loop
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", ncols=100)

    for i, batch in progress_bar:
        # Move inputs and labels to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights after a set number of steps (gradient accumulation)
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss


# Define Evaluation Loop with Progress Bar
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # Create a tqdm progress bar for the evaluation loop
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating", ncols=100)

        for i, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def save_model(model, tokenizer, save_path="./fine_tuned_model_lora"):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def save_model_as_pt(model, save_path="./fine_tuned_model/model.pt"):
    torch.save(model.state_dict(), save_path)


def load_model_from_pt(model_class, model_path, config_path):
    model = model_class.from_pretrained(config_path)
    model.load_state_dict(torch.load(model_path))
    return model


# 5. Training the Model
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_epoch(model, train_loader, optimizer, device, gradient_accumulation_steps)
    evaluate_model(model, val_loader, device)
    save_model(model, tokenizer, f"./fine_tuned_model_lora_epoch_{epoch + 1}")

# Save the Model
model.save_pretrained("./fine_tuned_model_lora")
tokenizer.save_pretrained("./fine_tuned_model_lora")
save_model_as_pt(model, f"./fine_tuned_model_lora/lora_model_epoch_{epoch + 1}.pt")
# torch.save(model, f"./fine_tuned_model_lora/lora_model.pt")
