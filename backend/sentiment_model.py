#!/usr/bin/env python
"""
Financial Sentiment Analysis using FinBERT

This script demonstrates how to load a pre-trained financial sentiment model
and analyze the sentiment of texts related to securities and exchange information.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_model(model_name="ProsusAI/finBERT"):
    """
    Load the tokenizer and model from Hugging Face model hub.
    FinBERT is specifically fine-tuned on financial texts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    """
    Analyze the sentiment of the input text.
    Returns the predicted sentiment label and associated probabilities.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and compute probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Define the sentiment labels (as used by FinBERT)
    labels = ["negative", "neutral", "positive"]
    
    # Get the predicted label index
    pred_index = int(np.argmax(probabilities))
    
    return labels[pred_index], probabilities

def main():
    # Example text related to securities/exchange information
    example_texts = [
        "PiScan.io has just launched the Account Statistics feature, allowing you to explore the Top 10,000 accounts with the highest balances and view the overall distribution of over 12 million accounts by tier.",
        "Uncertainty looms as regulatory changes affect the securities market adversely.",
        "Investors remain cautious despite a mixed outlook on economic growth."
    ]
    
    # Load the FinBERT model and tokenizer
    tokenizer, model = load_model("ProsusAI/finBERT")  # FinBERT model from ProsusAI :contentReference[oaicite:3]{index=3}
    
    print("Financial Sentiment Analysis using FinBERT")
    print("-----------------------------------------")
    
    # Analyze each example text
    for text in example_texts:
        sentiment, probs = analyze_sentiment(text, tokenizer, model)
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment}")
        print("Probabilities: " + ", ".join([f"{label}: {prob:.3f}" for label, prob in zip(["negative", "neutral", "positive"], probs)]))

if __name__ == "__main__":
    main()
