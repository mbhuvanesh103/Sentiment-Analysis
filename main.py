import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import re
from tqdm import tqdm 
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
def load_data(file_path):
    # Try common encodings
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")  # fallback
    # or you can use encoding="ISO-8859-1"
    
    # Clean text data
    def clean_text(text):
        text = re.sub(r'http\S+', '', str(text))
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    df = df.dropna(subset=['cleaned_text', 'sentiment_label'])
    
    return df


# Create a custom dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function
from tqdm import tqdm  # add this at the top

def train_model(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    # Wrap data_loader with tqdm
    loop = tqdm(data_loader, leave=True)
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Update tqdm description
        loop.set_description(f"Training Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0
    
    loop = tqdm(data_loader, leave=True)
    
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            
            loop.set_description(f"Eval Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average='weighted'
    )
    
    return avg_loss, accuracy, precision, recall, f1


# Main function
def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('test.csv')
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['cleaned_text'].tolist(), 
        df['sentiment_label'].tolist(), 
        test_size=0.3, 
        random_state=42,
        stratify=df['sentiment_label']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_len)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        
        start_time = time.time()
        train_loss = train_model(model, train_loader, optimizer, scheduler, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, device
        )
        epoch_time = time.time() - start_time
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'epoch_time': epoch_time
        })
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        print()
    
    # Load the best model and evaluate on test set
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        model, test_loader, device
    )
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # Save training statistics to a file
    stats_df = pd.DataFrame(training_stats)
    stats_df.to_csv('training_stats.csv', index=False)
    
    # Example prediction
    def predict_sentiment(text, model, tokenizer, device, max_len=128):
        model.eval()
        
        cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)
        cleaned_text = ' '.join(cleaned_text.lower().split())
        
        encoding = tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)
        
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return sentiment_map[prediction.item()]
    
    # Test with some examples
    test_samples = [
        "I love this product! It's amazing!",
        "This is okay, nothing special.",
        "I hate this, it's terrible."
    ]
    
    print("\nSample predictions:")
    for sample in test_samples:
        prediction = predict_sentiment(sample, model, tokenizer, device)
        print(f"Text: {sample}")
        print(f"Predicted sentiment: {prediction}\n")

if __name__ == "__main__":
    main()
