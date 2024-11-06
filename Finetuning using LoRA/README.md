# 🍅 Tomato Leaf Disease Classification with LoRA Fine-tuning

A PyTorch-based deep learning project that classifies tomato leaf diseases using custom model architecture and demonstrates the effectiveness of LoRA (Low-Rank Adaptation) fine-tuning for improving classification accuracy on underrepresented classes.

## 📊 Project Overview

This project focuses on:
1. Building a custom PyTorch model for tomato leaf disease classification
2. Addressing class imbalance issues
3. Implementing LoRA fine-tuning to improve accuracy for specific disease classes

## 🔑 Key Features

- Custom PyTorch model architecture
- LoRA fine-tuning implementation
- Handling class imbalance
- Disease-specific accuracy improvements
- Easy-to-use inference pipeline

## 📈 Results

### Initial Training
- Overall Model Accuracy: 74.4%
- Bacterial Spot Classification Accuracy: 32%
- Training Duration: 10 epochs

### After LoRA Fine-tuning
- Improved Overall Accuracy: 76.1%
- Bacterial Spot Classification Accuracy: 49%
- Significant improvement in underrepresented class performance

## 🛠️ Technical Details

### Dataset
- Multiple tomato leaf disease classes
- Intentionally reduced samples for `Tomato___Bacterial_spot` class to simulate real-world class imbalance

### Model Architecture
- Custom PyTorch implementation
- Optimized for leaf disease classification
- Adaptable for LoRA fine-tuning

### Fine-tuning Strategy
- Focused LoRA fine-tuning on `Tomato___Bacterial_spot` class
- Maintained model performance on other classes while improving target class accuracy

## 📊 Model Performance

| Stage | Overall Accuracy | Bacterial Spot Accuracy |
|-------|-----------------|----------------------|
| Initial Training | 74.4% | 32% |
| After LoRA Fine-tuning | 76.1% | 49% |
