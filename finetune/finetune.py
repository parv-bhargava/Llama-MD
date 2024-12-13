import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import login
from tqdm import tqdm  # Import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot")

# Preprocess Dataset
def preprocess_data(example):
    input_text = f"{example['Description']} Patient: {example['Patient']}"
    output_text = example["Doctor"]
    return {"input_text": input_text, "output_text": output_text}

processed_dataset = dataset.map(preprocess_data)

login("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") # Put your huggingface token here

# Tokenizer and Data Preparation
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|end_of_text|>'})

def tokenize_data(example):
    inputs = tokenizer(
        example["input_text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    outputs = tokenizer(
        example["output_text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
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
        "labels": labels
    }

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# 3. Load Model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=False)
model.to(device)

# Freeze Layers
for name, param in model.named_parameters():
    # Freeze the first few layers (example: freezing all layers up to 'transformer.h.10')
    if name.split('.')[2] < str(10):
        param.requires_grad = False
    else:
        param.requires_grad = True

# Verify that the layers have been frozen
for name, param in model.named_parameters():
    print(f"Layer {name} - Requires Grad: {param.requires_grad}")

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

        for batch in progress_bar:
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

# 5. Training the Model
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_epoch(model, train_loader, optimizer, device, gradient_accumulation_steps)
    evaluate_model(model, val_loader, device)

# Save the Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Example Inference
def generate_response(input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=200, num_beams=5)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

example_input = "What are the symptoms of flu? Patient: I have a fever and a cough."
print("Response:", generate_response(example_input))
