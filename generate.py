import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

# Step 1: Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad_token to eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 2: Load dataset (Wikitext in this example) and preprocess it
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Custom collate function to pad sequences in each batch
def collate_fn(batch):
    # Convert the list of lists into tensors
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    attention_masks = [torch.tensor(example['attention_mask']) for example in batch]
    
    # Pad the sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {'input_ids': input_ids, 'attention_mask': attention_masks}

# Create DataLoader with custom collate_fn
train_dataset = tokenized_datasets["train"]
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Step 3: Define the training function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

def train(model, train_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item()}")

# Step 4: Train the model
for epoch in range(3):  # Number of epochs to train
    print(f"Epoch {epoch + 1}")
    train(model, train_loader, optimizer, device)

    # Save the entire model



torch.save(model, "gpt2_finetuned_model.pth")

# Step 5: Generate text using the trained model
model.eval()

input_prompt = "Once upon a time, "
input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Text:")
print(generated_text)
