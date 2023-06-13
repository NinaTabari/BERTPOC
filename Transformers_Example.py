from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch

# Initialize the tokenizer with a pretrained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode our training data
def encode_data(tokenizer, texts, labels, max_length=256):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    inputs = {
        'input_ids': torch.tensor(encodings['input_ids']),
        'attention_mask': torch.tensor(encodings['attention_mask']),
        'labels': torch.tensor(labels)
    }
    return inputs

# Assume you have your texts and labels stored in these variables
texts = ["Hello, world!", "Another text sample"]
labels = [0, 1]

inputs = encode_data(tokenizer, texts, labels)

# Create a DataLoader for the training data
train_data = DataLoader(inputs, sampler=RandomSampler(inputs), batch_size=16)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.train()

# Define the training parameters
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 3

# Train the model
for epoch in range(epochs):
    total_loss = 0
    for batch in train_data:
        # Get the inputs from the batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {total_loss}')

# Save the model
model.save_pretrained("path/to/save/directory")
tokenizer.save_pretrained("path/to/save/directory")
