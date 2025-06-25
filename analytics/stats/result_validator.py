"""
Result Validator for Statistics System.

Validates calculation results for consistency, data quality, and format.
"""

import logging
from typing import Dict, Any, List, Union
import json
import math
from datetime import datetime


class ResultValidator:
    """
    Validates statistic calculation results.
    
    Features:
    - Schema validation for result structure
    - Data type and range validation
    - Statistical outlier detection
    - Format standardization
    """
    
    def __init__(self, config):
        """
        Initialize result validator.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.max_value_threshold = config.get_float(
            'STATISTICS_MAX_VALUE_THRESHOLD', 1e6)
        self.min_value_threshold = config.get_float(
            'STATISTICS_MIN_VALUE_THRESHOLD', -1e6)
        
        self.logger.info("ResultValidator initialized")
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize a calculation result.
        
        Parameters:
        -----------
        result : Dict[str, Any]
            Raw calculation result
            
        Returns:
        --------
        Dict[str, Any]
            Validated and standardized result
            
        Raises:
        -------
        ValueError
            If result is invalid or cannot be fixed
        """
        if not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")
        
        # Create validated copy
        validated = result.copy()
        
        # Validate required fields
        self._validate_required_fields(validated)
        
        # Validate data types
        self._validate_data_types(validated)
        
        # Validate statistical values
        self._validate_statistical_values(validated)
        
        # Standardize format
        self._standardize_format(validated)
        
        # Add validation metadata
        validated['validation_timestamp'] = datetime.now()
        validated['validation_status'] = 'passed'
        
        return validated
    
    def _validate_required_fields(self, result: Dict[str, Any]) -> None:
        """Validate that required fields are present."""
        required_fields = [
            'statistic_name',
            'statistic_value',
            'success'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in result:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate nested required fields in metadata
        if 'statistic_name' in result:
            metadata = result.get('metadata', {})
            if not metadata.get('description'):
                self.logger.warning(
                    f"Missing description for {result['statistic_name']}")
    
    def _validate_data_types(self, result: Dict[str, Any]) -> None:
        """Validate data types of result fields."""
        type_validations = {
            'statistic_name': str,
            'success': bool,
            'statistic_value': (int, float, dict, list, str, type(None)),
            'execution_time_ms': (int, float, type(None)),
            'data_points_used': (int, type(None))
        }
        
        for field, expected_types in type_validations.items():
            if field in result:
                value = result[field]
                if not isinstance(value, expected_types):
                    # Try to convert if possible
                    try:
                        if field == 'success':
                            result[field] = bool(value)
                        elif field in ['execution_time_ms', 'data_points_used']:
                            if value is not None:
                                result[field] = float(value) if '.' in str(
                                    value) else int(value)
                        elif field == 'statistic_name':
                            result[field] = str(value)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Invalid type for {field}: expected "
                            f"{expected_types}, got {type(value)}"
                        )
    
    def _validate_statistical_values(self, result: Dict[str, Any]) -> None:
        """Validate statistical values for sanity."""
        value = result.get('statistic_value')
        statistic_name = result.get('statistic_name', 'unknown')
        
        if value is None:
            return
        
        # Validate numeric values
        if isinstance(value, (int, float)):
            self._validate_numeric_value(value, statistic_name)
        
        # Validate arrays/lists
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (int, float)):
                    self._validate_numeric_value(item, f"{statistic_name}[{i}]")
        
        # Validate dictionaries (structured results)
        elif isinstance(value, dict):
            self._validate_dict_values(value, statistic_name)
    
    def _validate_numeric_value(self, value: Union[int, float],
                                context: str) -> None:
        """Validate a single numeric value."""
        # Check for NaN or infinity
        if math.isnan(value):
            raise ValueError(f"NaN value detected in {context}")
        
        if math.isinf(value):
            raise ValueError(f"Infinite value detected in {context}")
        
        # Check reasonable bounds
        if value > self.max_value_threshold:
            self.logger.warning(
                f"Large value detected in {context}: {value}")
        
        if value < self.min_value_threshold:
            self.logger.warning(
                f"Very negative value detected in {context}: {value}")
        
        # Statistic-specific validations
        self._validate_statistic_specific(value, context)
    
    def _validate_statistic_specific(self, value: Union[int, float],
                                     context: str) -> None:
        """Apply statistic-specific validation rules."""
        context_lower = context.lower()
        
        # Probability values should be between 0 and 1
        if any(term in context_lower for term in ['p_value', 'probability',
                                                  'confidence']):
            if not (0 <= value <= 1):
                raise ValueError(
                    f"Probability value out of range [0,1]: {value} in {context}")
        
        # Correlation values should be between -1 and 1
        if 'correlation' in context_lower:
            if not (-1 <= value <= 1):
                raise ValueError(
                    f"Correlation value out of range [-1,1]: {value} "
                    f"in {context}")
        
        # R-squared values should be between 0 and 1
        if 'r_squared' in context_lower or 'r2' in context_lower:
            if not (0 <= value <= 1):
                raise ValueError(
                    f"R-squared value out of range [0,1]: {value} in {context}")
        
        # Volatility/standard deviation should be positive
        if any(term in context_lower for term in ['volatility', 'std', 'sigma']):
            if value < 0:
                raise ValueError(
                    f"Volatility/std should be non-negative: {value} "
                    f"in {context}")
        
        # Half-life should be positive
        if 'half_life' in context_lower:
            if value <= 0:
                raise ValueError(
                    f"Half-life should be positive: {value} in {context}")
    
    def _validate_dict_values(self, value_dict: Dict[str, Any],
                              context: str) -> None:
        """Validate values within a dictionary result."""
        for key, val in value_dict.items():
            if isinstance(val, (int, float)):
                self._validate_numeric_value(val, f"{context}.{key}")
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, (int, float)):
                        self._validate_numeric_value(
                            item, f"{context}.{key}[{i}]")
            elif isinstance(val, dict):
                self._validate_dict_values(val, f"{context}.{key}")
    
    def _standardize_format(self, result: Dict[str, Any]) -> None:
        """Standardize result format and add missing fields."""
        # Ensure metadata exists
        if 'metadata' not in result:
            result['metadata'] = {}
        
        # Standardize success field
        if 'error' in result and result.get('success', True):
            result['success'] = False
        
        # Add execution time if missing
        if 'execution_time_ms' not in result:
            result['execution_time_ms'] = None
        
        # Ensure statistic_value is JSON serializable
        if 'statistic_value' in result:
            try:
                json.dumps(result['statistic_value'])
            except TypeError:
                # Convert non-serializable types to string representation
                result['statistic_value'] = str(result['statistic_value'])
                self.logger.warning(
                    f"Converted non-serializable value to string for "
                    f"{result.get('statistic_name', 'unknown')}"
                )
        
        # Extract metadata from result if available
        self._extract_metadata(result)
    
    def _extract_metadata(self, result: Dict[str, Any]) -> None:
        """Extract and organize metadata from result."""
        metadata = result.get('metadata', {})
        
        # Extract common metadata fields
        if 'statistic_name' in result and 'name' not in metadata:
            metadata['name'] = result['statistic_name']
        
        # Add validation info
        metadata['validated'] = True
        metadata['validator_version'] = '1.0.0'
        
        result['metadata'] = metadata
    
    def validate_batch_results(self, results: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Validate a batch of results.
        
        Parameters:
        -----------
        results : List[Dict[str, Any]]
            List of calculation results
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of validated results (invalid ones are filtered out)
        """
        validated_results = []
        validation_summary = {
            'total_results': len(results),
            'valid_results': 0,
            'invalid_results': 0,
            'errors': []
        }
        
        for i, result in enumerate(results):
            try:
                validated_result = self.validate_result(result)
                validated_results.append(validated_result)
                validation_summary['valid_results'] += 1
            except Exception as e:
                validation_summary['invalid_results'] += 1
                validation_summary['errors'].append({
                    'index': i,
                    'statistic_name': result.get('statistic_name', 'unknown'),
                    'error': str(e)
                })
                self.logger.warning(
                    f"Result validation failed for index {i}: {e}")
        
        # Log validation summary
        self.logger.info(
            f"Batch validation completed: "
            f"{validation_summary['valid_results']} valid, "
            f"{validation_summary['invalid_results']} invalid"
        )
        
        if validation_summary['invalid_results'] > 0:
            self.logger.warning(
                f"Validation errors: {validation_summary['errors']}")
        
        return validated_results
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules and thresholds."""
        return {
            'max_value_threshold': self.max_value_threshold,
            'min_value_threshold': self.min_value_threshold,
            'required_fields': [
                'statistic_name', 'statistic_value', 'success'],
            'statistic_specific_rules': {
                'probability_range': [0, 1],
                'correlation_range': [-1, 1],
                'r_squared_range': [0, 1],
                'volatility_min': 0,
                'half_life_min': 0
            }
        }
