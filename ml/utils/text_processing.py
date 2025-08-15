"""
Text Processing Utilities for MindGuard

This module provides text cleaning and preprocessing functionality
for mental health crisis detection.
"""

import re
import string
import logging
from typing import List, Optional, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing utilities for mental health data"""
    
    def __init__(self, language: str = "english"):
        self.language = language
        self._initialize_nltk()
        self._initialize_spacy()
        
        # Mental health specific patterns
        self.crisis_keywords = {
            'suicide': ['kill myself', 'end it all', 'want to die', 'suicide', 'self harm'],
            'depression': ['depressed', 'hopeless', 'worthless', 'no reason to live'],
            'anxiety': ['panic', 'anxiety', 'overwhelmed', 'can\'t breathe'],
            'crisis': ['help me', 'emergency', 'crisis', 'urgent']
        }
        
        # Reddit-specific patterns
        self.reddit_patterns = {
            'subreddit_mentions': r'r/\w+',
            'user_mentions': r'u/\w+',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'markdown': r'[*_`~#]+',
        }
    
    def _initialize_nltk(self):
        """Initialize NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()
    
    def _initialize_spacy(self):
        """Initialize spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(self.reddit_patterns['urls'], '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(self.reddit_patterns['subreddit_mentions'], '', text)
        text = re.sub(self.reddit_patterns['user_mentions'], '', text)
        text = re.sub(self.reddit_patterns['markdown'], '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (keep apostrophes for contractions)
        text = re.sub(r'[^\w\s\']', '', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stop words from text"""
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text"""
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def detect_sentiment(self, text: str) -> Dict[str, float]:
        """Detect sentiment using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_crisis_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract crisis-related keywords and phrases"""
        text_lower = text.lower()
        indicators = {}
        
        for category, keywords in self.crisis_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                indicators[category] = found_keywords
        
        return indicators
    
    def normalize_text(self, text: str) -> str:
        """Complete text normalization pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Remove stop words
        text = self.remove_stopwords(text)
        
        # Lemmatize
        text = self.lemmatize_text(text)
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive text features"""
        features = {}
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Sentiment features
        sentiment = self.detect_sentiment(text)
        features['sentiment_polarity'] = sentiment['polarity']
        features['sentiment_subjectivity'] = sentiment['subjectivity']
        
        # Crisis indicators
        crisis_indicators = self.extract_crisis_indicators(text)
        features['crisis_indicators'] = crisis_indicators
        
        # Entity features
        entities = self.extract_entities(text)
        features['entity_count'] = len(entities)
        features['entities'] = entities
        
        # Readability features
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        return features
    
    def preprocess_for_ml(self, text: str, max_length: int = 512) -> str:
        """Preprocess text specifically for ML models"""
        # Clean and normalize
        text = self.normalize_text(text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def batch_process(self, texts: List[str]) -> List[str]:
        """Process a batch of texts"""
        return [self.preprocess_for_ml(text) for text in texts]


class CrisisTextAnalyzer:
    """Specialized analyzer for crisis-related text"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        
        # Crisis severity patterns
        self.severity_patterns = {
            'immediate': [
                r'i want to die',
                r'i\'m going to kill myself',
                r'i can\'t take it anymore',
                r'goodbye',
                r'this is my last message'
            ],
            'high': [
                r'i feel hopeless',
                r'i want to end it all',
                r'i have no reason to live',
                r'i\'m worthless'
            ],
            'moderate': [
                r'i\'m depressed',
                r'i feel sad',
                r'i need help',
                r'i\'m struggling'
            ]
        }
    
    def assess_crisis_severity(self, text: str) -> Dict[str, Any]:
        """Assess the severity of crisis indicators in text"""
        text_lower = text.lower()
        severity_scores = {'immediate': 0, 'high': 0, 'moderate': 0}
        
        for severity, patterns in self.severity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    severity_scores[severity] += 1
        
        # Determine overall severity
        if severity_scores['immediate'] > 0:
            overall_severity = 'immediate'
        elif severity_scores['high'] > 0:
            overall_severity = 'high'
        elif severity_scores['moderate'] > 0:
            overall_severity = 'moderate'
        else:
            overall_severity = 'low'
        
        return {
            'severity': overall_severity,
            'scores': severity_scores,
            'indicators_found': sum(severity_scores.values())
        }
    
    def extract_urgency_indicators(self, text: str) -> List[str]:
        """Extract urgency indicators from text"""
        urgency_patterns = [
            r'right now',
            r'immediately',
            r'urgent',
            r'emergency',
            r'asap',
            r'now',
            r'today'
        ]
        
        indicators = []
        for pattern in urgency_patterns:
            if re.search(pattern, text.lower()):
                indicators.append(pattern)
        
        return indicators
