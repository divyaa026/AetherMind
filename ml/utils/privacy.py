"""
Privacy Processing Utilities for MindGuard

This module provides privacy protection and PII handling for mental health data.
"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass
import numpy as np
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


@dataclass
class PIIPattern:
    """PII pattern definition"""
    name: str
    pattern: str
    replacement: str
    description: str


class PrivacyProcessor:
    """Privacy protection utilities for mental health data"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # PII patterns for detection and removal
        self.pii_patterns = [
            PIIPattern(
                name="email",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement="[EMAIL]",
                description="Email addresses"
            ),
            PIIPattern(
                name="phone",
                pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                replacement="[PHONE]",
                description="Phone numbers"
            ),
            PIIPattern(
                name="ssn",
                pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                replacement="[SSN]",
                description="Social Security Numbers"
            ),
            PIIPattern(
                name="credit_card",
                pattern=r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
                replacement="[CREDIT_CARD]",
                description="Credit card numbers"
            ),
            PIIPattern(
                name="ip_address",
                pattern=r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                replacement="[IP_ADDRESS]",
                description="IP addresses"
            ),
            PIIPattern(
                name="url",
                pattern=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                replacement="[URL]",
                description="URLs"
            ),
            PIIPattern(
                name="username",
                pattern=r'@\w+',
                replacement="[USERNAME]",
                description="Usernames/mentions"
            ),
            PIIPattern(
                name="address",
                pattern=r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                replacement="[ADDRESS]",
                description="Street addresses"
            ),
            PIIPattern(
                name="date_of_birth",
                pattern=r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-]\d{4}\b',
                replacement="[DATE]",
                description="Dates of birth"
            ),
            PIIPattern(
                name="zip_code",
                pattern=r'\b\d{5}(?:-\d{4})?\b',
                replacement="[ZIP_CODE]",
                description="ZIP codes"
            )
        ]
        
        # Mental health specific privacy patterns
        self.mental_health_patterns = [
            PIIPattern(
                name="hospital_name",
                pattern=r'\b(?:hospital|clinic|medical center|health center)\s+[A-Za-z\s]+\b',
                replacement="[HEALTHCARE_FACILITY]",
                description="Healthcare facility names"
            ),
            PIIPattern(
                name="doctor_name",
                pattern=r'Dr\.\s+[A-Za-z\s]+',
                replacement="[HEALTHCARE_PROVIDER]",
                description="Doctor names"
            ),
            PIIPattern(
                name="medication",
                pattern=r'\b(?:Prozac|Zoloft|Lexapro|Paxil|Celexa|Wellbutrin|Effexor|Cymbalta|Abilify|Risperdal|Seroquel|Zyprexa)\b',
                replacement="[MEDICATION]",
                description="Common psychiatric medications"
            )
        ]
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        detected_pii = {}
        
        for pattern in self.pii_patterns + self.mental_health_patterns:
            matches = re.findall(pattern.pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pattern.name] = matches
        
        return detected_pii
    
    def remove_pii(self, text: str) -> str:
        """Remove PII from text"""
        if not text or not isinstance(text, str):
            return text
        
        cleaned_text = text
        
        # Apply all PII patterns
        for pattern in self.pii_patterns + self.mental_health_patterns:
            cleaned_text = re.sub(pattern.pattern, pattern.replacement, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def anonymize_text(self, text: str, anonymization_level: str = "standard") -> str:
        """Anonymize text with different levels of privacy protection"""
        if not text or not isinstance(text, str):
            return text
        
        if anonymization_level == "minimal":
            # Only remove obvious PII
            return self.remove_pii(text)
        
        elif anonymization_level == "standard":
            # Remove PII and some identifying information
            text = self.remove_pii(text)
            
            # Remove specific locations (cities, states)
            text = re.sub(r'\b[A-Z][a-z]+(?:,\s*[A-Z]{2})?\b', '[LOCATION]', text)
            
            # Remove specific ages
            text = re.sub(r'\b(?:I am|I\'m)\s+\d+\s+(?:years old|yo)\b', '[AGE]', text)
            
            return text
        
        elif anonymization_level == "strict":
            # Maximum privacy protection
            text = self.remove_pii(text)
            
            # Remove all names (basic pattern)
            text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME]', text)
            
            # Remove specific numbers
            text = re.sub(r'\b\d+\b', '[NUMBER]', text)
            
            # Remove specific locations
            text = re.sub(r'\b[A-Z][a-z]+(?:,\s*[A-Z]{2})?\b', '[LOCATION]', text)
            
            return text
        
        else:
            return self.remove_pii(text)
    
    def hash_identifier(self, identifier: str, salt: Optional[str] = None) -> str:
        """Hash an identifier with optional salt"""
        if salt:
            identifier = identifier + salt
        
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return data
        
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return encrypted_data
    
    def apply_k_anonymity(self, data: List[Dict[str, Any]], k: int = 50, 
                         quasi_identifiers: List[str] = None) -> List[Dict[str, Any]]:
        """Apply k-anonymity to dataset"""
        if not quasi_identifiers:
            quasi_identifiers = ['age', 'location', 'gender']
        
        # This is a simplified implementation
        # In practice, you would use more sophisticated k-anonymity algorithms
        
        anonymized_data = []
        
        for record in data:
            anonymized_record = record.copy()
            
            for identifier in quasi_identifiers:
                if identifier in anonymized_record:
                    # Generalize the value
                    value = anonymized_record[identifier]
                    if isinstance(value, (int, float)):
                        # Generalize numeric values
                        anonymized_record[identifier] = f"[{identifier.upper()}_RANGE]"
                    else:
                        # Generalize categorical values
                        anonymized_record[identifier] = f"[{identifier.upper()}_CATEGORY]"
            
            anonymized_data.append(anonymized_record)
        
        return anonymized_data
    
    def generate_privacy_report(self, text: str) -> Dict[str, Any]:
        """Generate a privacy analysis report for text"""
        detected_pii = self.detect_pii(text)
        
        report = {
            "text_length": len(text),
            "pii_detected": len(detected_pii) > 0,
            "pii_types": list(detected_pii.keys()),
            "pii_count": sum(len(matches) for matches in detected_pii.values()),
            "privacy_risk_level": self._assess_privacy_risk(detected_pii),
            "recommendations": self._generate_privacy_recommendations(detected_pii)
        }
        
        return report
    
    def _assess_privacy_risk(self, detected_pii: Dict[str, List[str]]) -> str:
        """Assess privacy risk level based on detected PII"""
        high_risk_patterns = ['ssn', 'credit_card', 'email']
        medium_risk_patterns = ['phone', 'address', 'date_of_birth']
        
        high_risk_count = sum(1 for pattern in high_risk_patterns if pattern in detected_pii)
        medium_risk_count = sum(1 for pattern in medium_risk_patterns if pattern in detected_pii)
        
        if high_risk_count > 0:
            return "HIGH"
        elif medium_risk_count > 2:
            return "MEDIUM"
        elif len(detected_pii) > 0:
            return "LOW"
        else:
            return "NONE"
    
    def _generate_privacy_recommendations(self, detected_pii: Dict[str, List[str]]) -> List[str]:
        """Generate privacy protection recommendations"""
        recommendations = []
        
        if 'ssn' in detected_pii:
            recommendations.append("Remove Social Security Numbers immediately")
        
        if 'credit_card' in detected_pii:
            recommendations.append("Remove credit card numbers immediately")
        
        if 'email' in detected_pii:
            recommendations.append("Consider anonymizing email addresses")
        
        if 'phone' in detected_pii:
            recommendations.append("Consider anonymizing phone numbers")
        
        if 'address' in detected_pii:
            recommendations.append("Consider generalizing addresses")
        
        if len(detected_pii) > 5:
            recommendations.append("High PII density detected - consider strict anonymization")
        
        if not recommendations:
            recommendations.append("No immediate privacy concerns detected")
        
        return recommendations


class DifferentialPrivacy:
    """Differential privacy utilities for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float) -> float:
        """Add Gaussian noise for differential privacy"""
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def clip_gradients(self, gradients: List[float], clip_norm: float) -> List[float]:
        """Clip gradients for differential privacy"""
        total_norm = np.sqrt(sum(g**2 for g in gradients))
        clip_coef = min(clip_norm / total_norm, 1.0)
        return [g * clip_coef for g in gradients]


class HomomorphicEncryption:
    """Homomorphic encryption utilities for secure computation"""
    
    def __init__(self):
        # This is a simplified implementation
        # In practice, you would use a proper homomorphic encryption library
        self.public_key = None
        self.private_key = None
    
    def encrypt(self, value: float) -> bytes:
        """Encrypt a value for homomorphic computation"""
        # Simplified implementation
        # In practice, use libraries like PySEAL or TenSEAL
        return str(value).encode()
    
    def decrypt(self, encrypted_value: bytes) -> float:
        """Decrypt a value from homomorphic computation"""
        # Simplified implementation
        return float(encrypted_value.decode())
    
    def add_encrypted(self, encrypted_a: bytes, encrypted_b: bytes) -> bytes:
        """Add two encrypted values"""
        # Simplified implementation
        a = self.decrypt(encrypted_a)
        b = self.decrypt(encrypted_b)
        return self.encrypt(a + b)
    
    def multiply_encrypted(self, encrypted_a: bytes, encrypted_b: bytes) -> bytes:
        """Multiply two encrypted values"""
        # Simplified implementation
        a = self.decrypt(encrypted_a)
        b = self.decrypt(encrypted_b)
        return self.encrypt(a * b)
