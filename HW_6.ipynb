from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import pandas as pd

ds = load_dataset("xguman/hw5_text_dataset")

tokenizer = AutoTokenizer.from_pretrained("classla/xlm-roberta-base-multilingual-text-genre-classifier")
model = AutoModelForSequenceClassification.from_pretrained("classla/xlm-roberta-base-multilingual-text-genre-classifier")

model.eval()

id2label = model.config.id2label

def predict_genre(batch):
    inputs = tokenizer(batch['text'], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).tolist()
    batch['predicted_genre'] = predicted_class
    return batch

ds_with_predictions = ds.map(predict_genre, batched=True, batch_size=16)

df = pd.DataFrame(ds_with_predictions['train'])

df['predicted_genre_label'] = df['predicted_genre'].apply(lambda x: id2label[x])

print(df[['text', 'predicted_genre', 'predicted_genre_label']].head(10))