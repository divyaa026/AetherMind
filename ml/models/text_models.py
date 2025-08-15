"""
Text Analysis Models for MindGuard

This module contains text-based crisis detection models including BERT and BiLSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for text models"""
    model_name: str
    max_length: int = 512
    hidden_size: int = 768
    num_classes: int = 2
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BERTCrisisDetector(nn.Module):
    """BERT-based crisis detection model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers for fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for fine-tuning
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        
        # Crisis-specific attention layer
        self.crisis_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass"""
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get sequence output
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply crisis-specific attention
        attended_output, _ = self.crisis_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=(attention_mask == 0)
        )
        
        # Add residual connection and layer norm
        sequence_output = self.layer_norm(sequence_output + attended_output)
        
        # Global average pooling
        pooled_output = torch.mean(sequence_output, dim=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_crisis_probability(self, text: str, tokenizer: BertTokenizer) -> Dict[str, Any]:
        """Predict crisis probability for a single text"""
        self.eval()
        
        # Tokenize text
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.forward(**inputs)
            probabilities = F.softmax(logits, dim=1)
            crisis_prob = probabilities[0, 1].item()
        
        return {
            "crisis_probability": crisis_prob,
            "confidence": max(probabilities[0]).item(),
            "prediction": "crisis" if crisis_prob > 0.5 else "non_crisis"
        }
    
    def extract_crisis_features(self, text: str, tokenizer: BertTokenizer) -> Dict[str, Any]:
        """Extract crisis-related features from text"""
        self.eval()
        
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get BERT outputs
            bert_outputs = self.bert(**inputs)
            sequence_output = bert_outputs.last_hidden_state
            
            # Get attention weights
            attended_output, attention_weights = self.crisis_attention(
                sequence_output, sequence_output, sequence_output,
                key_padding_mask=(inputs['attention_mask'] == 0)
            )
            
            # Extract features
            features = {
                "bert_embeddings": sequence_output[0].cpu().numpy(),
                "attention_weights": attention_weights[0].cpu().numpy(),
                "crisis_attention": attended_output[0].cpu().numpy()
            }
        
        return features


class BiLSTMCrisisDetector(nn.Module):
    """BiLSTM-based crisis detection model"""
    
    def __init__(self, config: ModelConfig, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=config.hidden_size // 2,  # Bidirectional
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if 2 > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout
        )
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        
        # Crisis-specific features
        self.crisis_feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 64)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Pack sequence for LSTM
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            packed_embedded = embedded
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        if attention_mask is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
        )
        
        # Global average pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(attended_output.size())
            pooled_output = (attended_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = torch.mean(attended_output, dim=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def extract_crisis_features(self, input_ids, attention_mask=None):
        """Extract crisis-related features"""
        embedded = self.embedding(input_ids)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            packed_embedded = embedded
        
        lstm_output, (hidden, cell) = self.lstm(packed_embedded)
        
        if attention_mask is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Extract crisis features
        crisis_features = self.crisis_feature_extractor(lstm_output)
        
        return {
            "lstm_output": lstm_output,
            "crisis_features": crisis_features,
            "hidden_states": hidden
        }


class CrisisTextProcessor:
    """Text processor specifically for crisis detection"""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        # Crisis-specific vocabulary
        self.crisis_vocab = {
            'suicide': ['kill', 'die', 'death', 'suicide', 'end', 'overdose'],
            'depression': ['depressed', 'sad', 'hopeless', 'worthless', 'empty'],
            'anxiety': ['anxious', 'panic', 'worry', 'fear', 'stress'],
            'crisis': ['help', 'emergency', 'crisis', 'urgent', 'now'],
            'self_harm': ['cut', 'hurt', 'pain', 'bleed', 'scar']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for crisis detection"""
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_crisis_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract crisis indicators from text"""
        text_lower = text.lower()
        indicators = {}
        
        for category, keywords in self.crisis_vocab.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                indicators[category] = found_keywords
        
        return indicators
    
    def tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize text for model input"""
        return self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )


class CrisisDetectionPipeline:
    """Complete pipeline for crisis detection"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.text_processor = CrisisTextProcessor()
        
        # Initialize models
        self.bert_model = BERTCrisisDetector(model_config)
        self.bilstm_model = BiLSTMCrisisDetector(model_config, vocab_size=30000)
        
        # Move models to device
        self.bert_model.to(model_config.device)
        self.bilstm_model.to(model_config.device)
    
    def predict_crisis(self, text: str) -> Dict[str, Any]:
        """Predict crisis probability using ensemble"""
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(text)
        
        # Extract crisis indicators
        indicators = self.text_processor.extract_crisis_indicators(processed_text)
        
        # Get predictions from both models
        bert_prediction = self.bert_model.predict_crisis_probability(
            processed_text, self.text_processor.tokenizer
        )
        
        # Tokenize for BiLSTM
        tokenized = self.text_processor.tokenize_text(processed_text)
        
        # Get BiLSTM prediction
        with torch.no_grad():
            bilstm_logits = self.bilstm_model(**tokenized)
            bilstm_probs = F.softmax(bilstm_logits, dim=1)
            bilstm_crisis_prob = bilstm_probs[0, 1].item()
        
        # Ensemble prediction (simple average)
        ensemble_prob = (bert_prediction['crisis_probability'] + bilstm_crisis_prob) / 2
        
        return {
            "ensemble_probability": ensemble_prob,
            "bert_probability": bert_prediction['crisis_probability'],
            "bilstm_probability": bilstm_crisis_prob,
            "crisis_indicators": indicators,
            "prediction": "crisis" if ensemble_prob > 0.5 else "non_crisis",
            "confidence": max(bert_prediction['confidence'], bilstm_probs.max().item())
        }
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features for analysis"""
        processed_text = self.text_processor.preprocess_text(text)
        
        # Get features from both models
        bert_features = self.bert_model.extract_crisis_features(
            processed_text, self.text_processor.tokenizer
        )
        
        tokenized = self.text_processor.tokenize_text(processed_text)
        bilstm_features = self.bilstm_model.extract_crisis_features(**tokenized)
        
        # Combine features
        combined_features = {
            "bert_embeddings": bert_features["bert_embeddings"],
            "bert_attention": bert_features["attention_weights"],
            "bilstm_output": bilstm_features["lstm_output"].cpu().numpy(),
            "crisis_features": bilstm_features["crisis_features"].cpu().numpy(),
            "crisis_indicators": self.text_processor.extract_crisis_indicators(processed_text)
        }
        
        return combined_features
