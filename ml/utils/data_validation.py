"""
Data Validation Utilities for MindGuard

This module provides data validation and quality checks for mental health datasets.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Data validation rule"""
    name: str
    description: str
    check_function: callable
    severity: str = "error"  # error, warning, info


class DataValidator:
    """Data validation utilities for mental health datasets"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules"""
        rules = [
            ValidationRule(
                name="missing_values",
                description="Check for missing values in required columns",
                check_function=self._check_missing_values,
                severity="error"
            ),
            ValidationRule(
                name="text_length",
                description="Check text length constraints",
                check_function=self._check_text_length,
                severity="warning"
            ),
            ValidationRule(
                name="label_distribution",
                description="Check label distribution balance",
                check_function=self._check_label_distribution,
                severity="warning"
            ),
            ValidationRule(
                name="duplicate_texts",
                description="Check for duplicate text entries",
                check_function=self._check_duplicate_texts,
                severity="warning"
            ),
            ValidationRule(
                name="inappropriate_content",
                description="Check for inappropriate content",
                check_function=self._check_inappropriate_content,
                severity="error"
            ),
            ValidationRule(
                name="data_consistency",
                description="Check data consistency across columns",
                check_function=self._check_data_consistency,
                severity="error"
            )
        ]
        return rules
    
    def _check_missing_values(self, df: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Any]:
        """Check for missing values in required columns"""
        missing_text = df[text_column].isna().sum()
        missing_labels = df[label_column].isna().sum()
        
        return {
            "passed": missing_text == 0 and missing_labels == 0,
            "missing_text": missing_text,
            "missing_labels": missing_labels,
            "total_rows": len(df)
        }
    
    def _check_text_length(self, df: pd.DataFrame, text_column: str, min_length: int = 10, max_length: int = 2000) -> Dict[str, Any]:
        """Check text length constraints"""
        text_lengths = df[text_column].str.len()
        too_short = (text_lengths < min_length).sum()
        too_long = (text_lengths > max_length).sum()
        
        return {
            "passed": too_short == 0 and too_long == 0,
            "too_short": too_short,
            "too_long": too_long,
            "avg_length": text_lengths.mean(),
            "min_length": text_lengths.min(),
            "max_length": text_lengths.max()
        }
    
    def _check_label_distribution(self, df: pd.DataFrame, label_column: str, min_ratio: float = 0.1) -> Dict[str, Any]:
        """Check label distribution balance"""
        label_counts = df[label_column].value_counts()
        total_samples = len(df)
        
        imbalances = []
        for label, count in label_counts.items():
            ratio = count / total_samples
            if ratio < min_ratio:
                imbalances.append({
                    "label": label,
                    "count": count,
                    "ratio": ratio,
                    "min_required": min_ratio
                })
        
        return {
            "passed": len(imbalances) == 0,
            "label_distribution": label_counts.to_dict(),
            "imbalances": imbalances,
            "total_samples": total_samples
        }
    
    def _check_duplicate_texts(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for duplicate text entries"""
        duplicates = df[text_column].duplicated().sum()
        
        return {
            "passed": duplicates == 0,
            "duplicate_count": duplicates,
            "unique_texts": df[text_column].nunique(),
            "total_texts": len(df)
        }
    
    def _check_inappropriate_content(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for inappropriate content patterns"""
        inappropriate_patterns = [
            r'\b(spam|advertisement|promotion)\b',
            r'http[s]?://',
            r'@\w+',
            r'#\w+',
            r'[A-Z]{10,}',  # Excessive caps
        ]
        
        flagged_texts = []
        for pattern in inappropriate_patterns:
            matches = df[df[text_column].str.contains(pattern, case=False, na=False)]
            if not matches.empty:
                flagged_texts.extend(matches[text_column].tolist())
        
        return {
            "passed": len(flagged_texts) == 0,
            "flagged_count": len(set(flagged_texts)),
            "flagged_texts": list(set(flagged_texts))[:10]  # Limit to first 10
        }
    
    def _check_data_consistency(self, df: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Any]:
        """Check data consistency across columns"""
        issues = []
        
        # Check for empty strings
        empty_texts = df[df[text_column].str.strip() == ""]
        if not empty_texts.empty:
            issues.append(f"Found {len(empty_texts)} empty text entries")
        
        # Check for whitespace-only texts
        whitespace_only = df[df[text_column].str.strip().str.len() == 0]
        if not whitespace_only.empty:
            issues.append(f"Found {len(whitespace_only)} whitespace-only texts")
        
        # Check for non-string text entries
        non_string_texts = df[~df[text_column].apply(lambda x: isinstance(x, str))]
        if not non_string_texts.empty:
            issues.append(f"Found {len(non_string_texts)} non-string text entries")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def validate_dataset(self, df: pd.DataFrame, text_column: str, label_column: str, 
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive dataset validation"""
        logger.info("Starting dataset validation")
        
        if config is None:
            config = {
                "min_text_length": 10,
                "max_text_length": 2000,
                "min_label_ratio": 0.1
            }
        
        validation_results = {
            "dataset_info": {
                "total_rows": len(df),
                "columns": list(df.columns),
                "text_column": text_column,
                "label_column": label_column
            },
            "validation_results": {},
            "overall_passed": True,
            "errors": [],
            "warnings": []
        }
        
        # Run all validation rules
        for rule in self.validation_rules:
            try:
                if rule.name == "text_length":
                    result = rule.check_function(df, text_column, config["min_text_length"], config["max_text_length"])
                elif rule.name == "label_distribution":
                    result = rule.check_function(df, label_column, config["min_label_ratio"])
                else:
                    result = rule.check_function(df, text_column, label_column)
                
                validation_results["validation_results"][rule.name] = {
                    "description": rule.description,
                    "severity": rule.severity,
                    "result": result
                }
                
                if not result["passed"]:
                    if rule.severity == "error":
                        validation_results["overall_passed"] = False
                        validation_results["errors"].append(f"{rule.name}: {rule.description}")
                    else:
                        validation_results["warnings"].append(f"{rule.name}: {rule.description}")
                
            except Exception as e:
                logger.error(f"Error in validation rule {rule.name}: {e}")
                validation_results["validation_results"][rule.name] = {
                    "description": rule.description,
                    "severity": rule.severity,
                    "error": str(e)
                }
                validation_results["overall_passed"] = False
                validation_results["errors"].append(f"{rule.name}: Validation failed - {e}")
        
        logger.info(f"Dataset validation completed. Passed: {validation_results['overall_passed']}")
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate a detailed validation report"""
        report_lines = [
            "=" * 80,
            "MINDGUARD DATASET VALIDATION REPORT",
            "=" * 80,
            "",
            f"Dataset Information:",
            f"  Total Rows: {validation_results['dataset_info']['total_rows']}",
            f"  Columns: {', '.join(validation_results['dataset_info']['columns'])}",
            f"  Text Column: {validation_results['dataset_info']['text_column']}",
            f"  Label Column: {validation_results['dataset_info']['label_column']}",
            "",
            f"Overall Status: {'PASSED' if validation_results['overall_passed'] else 'FAILED'}",
            ""
        ]
        
        # Add validation results
        for rule_name, rule_result in validation_results["validation_results"].items():
            report_lines.extend([
                f"Rule: {rule_name}",
                f"  Description: {rule_result['description']}",
                f"  Severity: {rule_result['severity']}",
            ])
            
            if "error" in rule_result:
                report_lines.append(f"  Status: ERROR - {rule_result['error']}")
            else:
                status = "PASSED" if rule_result['result']['passed'] else "FAILED"
                report_lines.append(f"  Status: {status}")
                
                # Add detailed results
                for key, value in rule_result['result'].items():
                    if key != 'passed':
                        report_lines.append(f"    {key}: {value}")
            
            report_lines.append("")
        
        # Add errors and warnings
        if validation_results["errors"]:
            report_lines.extend([
                "ERRORS:",
                *[f"  - {error}" for error in validation_results["errors"]],
                ""
            ])
        
        if validation_results["warnings"]:
            report_lines.extend([
                "WARNINGS:",
                *[f"  - {warning}" for warning in validation_results["warnings"]],
                ""
            ])
        
        report_lines.extend([
            "=" * 80,
            "End of Report",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def suggest_fixes(self, validation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Suggest fixes for validation issues"""
        suggestions = {
            "data_cleaning": [],
            "data_balancing": [],
            "quality_improvements": []
        }
        
        for rule_name, rule_result in validation_results["validation_results"].items():
            if "error" in rule_result or not rule_result.get("result", {}).get("passed", True):
                if rule_name == "missing_values":
                    suggestions["data_cleaning"].append("Remove rows with missing values or impute them")
                
                elif rule_name == "text_length":
                    suggestions["data_cleaning"].append("Filter texts by length constraints")
                
                elif rule_name == "label_distribution":
                    suggestions["data_balancing"].append("Consider oversampling minority classes or undersampling majority classes")
                
                elif rule_name == "duplicate_texts":
                    suggestions["data_cleaning"].append("Remove duplicate text entries")
                
                elif rule_name == "inappropriate_content":
                    suggestions["quality_improvements"].append("Review and filter inappropriate content")
                
                elif rule_name == "data_consistency":
                    suggestions["data_cleaning"].append("Clean inconsistent data entries")
        
        return suggestions


class ClinicalDataValidator(DataValidator):
    """Specialized validator for clinical mental health data"""
    
    def __init__(self):
        super().__init__()
        self.clinical_rules = self._initialize_clinical_rules()
        self.validation_rules.extend(self.clinical_rules)
    
    def _initialize_clinical_rules(self) -> List[ValidationRule]:
        """Initialize clinical-specific validation rules"""
        return [
            ValidationRule(
                name="clinical_terminology",
                description="Check for appropriate clinical terminology",
                check_function=self._check_clinical_terminology,
                severity="warning"
            ),
            ValidationRule(
                name="risk_assessment",
                description="Check for risk assessment indicators",
                check_function=self._check_risk_assessment,
                severity="error"
            ),
            ValidationRule(
                name="confidentiality",
                description="Check for potential confidentiality breaches",
                check_function=self._check_confidentiality,
                severity="error"
            )
        ]
    
    def _check_clinical_terminology(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for appropriate clinical terminology"""
        clinical_terms = [
            'depression', 'anxiety', 'suicide', 'self-harm', 'crisis',
            'therapy', 'medication', 'psychiatrist', 'psychologist',
            'mental health', 'wellness', 'recovery'
        ]
        
        term_counts = {}
        for term in clinical_terms:
            count = df[df[text_column].str.contains(term, case=False, na=False)].shape[0]
            if count > 0:
                term_counts[term] = count
        
        return {
            "passed": len(term_counts) > 0,
            "clinical_terms_found": term_counts,
            "total_clinical_mentions": sum(term_counts.values())
        }
    
    def _check_risk_assessment(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for risk assessment indicators"""
        risk_indicators = [
            'suicide', 'kill myself', 'end it all', 'want to die',
            'self harm', 'cutting', 'overdose', 'no reason to live'
        ]
        
        risk_texts = []
        for indicator in risk_indicators:
            matches = df[df[text_column].str.contains(indicator, case=False, na=False)]
            if not matches.empty:
                risk_texts.extend(matches[text_column].tolist())
        
        return {
            "passed": len(risk_texts) > 0,  # Should have some risk indicators
            "risk_indicators_found": len(set(risk_texts)),
            "risk_texts": list(set(risk_texts))[:5]  # Limit to first 5
        }
    
    def _check_confidentiality(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for potential confidentiality breaches"""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
        ]
        
        pii_found = []
        for pattern in pii_patterns:
            matches = df[df[text_column].str.contains(pattern, na=False)]
            if not matches.empty:
                pii_found.extend(matches[text_column].tolist())
        
        return {
            "passed": len(pii_found) == 0,
            "pii_instances": len(set(pii_found)),
            "pii_texts": list(set(pii_found))[:3]  # Limit to first 3
        }
