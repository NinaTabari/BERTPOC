import torch
import csv
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('storylinemodel', from_tf=True) #replace 'model_dir' with your model directory


with open('data/validate.csv', 'r') as f:
    lines = f.read().splitlines()

inputs = tokenizer(lines, return_tensors='pt', padding=True, truncation=True)

model.eval()

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=-1).tolist()

def csv_to_dict(filename):
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None)  # Skip the headers
        mydict = {int(rows[0]):rows[1] for rows in reader}
    return mydict

# usage
mapdict = csv_to_dict('data/mapper.csv')

def print_predictions(filename):
    with open(filename, 'r') as file:
        predi = 0
        for line in file:
            print(line + "DEFECT:" + mapdict[predictions[predi]])
            predi = predi + 1


print_predictions('data/validate.csv')