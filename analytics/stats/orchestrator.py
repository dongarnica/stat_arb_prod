"""
Statistics Orchestrator - Manages module discovery and execution.
"""

import importlib
import inspect
import logging
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import pandas as pd

from .base_statistic import BaseStatistic
from .modules.pair_trading.base_pair_statistic import BasePairStatistic
from .configuration_manager import ConfigurationManager


class StatisticsOrchestrator:
    """
    Orchestrates the discovery, loading, and execution of statistic modules.
    """
    
    def __init__(self, config: ConfigurationManager):
        """Initialize the orchestrator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Module storage
        self.single_asset_modules: Dict[str, Type[BaseStatistic]] = {}
        self.pair_trading_modules: Dict[str, Type[BasePairStatistic]] = {}
        
        # Module metadata
        self.module_metadata: Dict[str, Dict[str, Any]] = {}
        
    def load_modules(self) -> int:
        """Load all available statistics modules."""
        total_loaded = 0
        
        # Module path
        modules_path = Path(__file__).parent / "modules"
        if not modules_path.exists():
            self.logger.warning(f"Modules directory not found: {modules_path}")
            return 0
        
        # Load single-asset modules
        single_asset_loaded = self._load_single_asset_modules(modules_path)
        total_loaded += single_asset_loaded
        
        # Load pair trading modules
        pair_trading_loaded = self._load_pair_trading_modules(modules_path)
        total_loaded += pair_trading_loaded
        
        self.logger.info(
            f"Module loading completed: {total_loaded} total "
            f"({single_asset_loaded} single-asset, "
            f"{pair_trading_loaded} pair-trading)"
        )
        
        return total_loaded
    
    def _load_single_asset_modules(self, modules_path: Path) -> int:
        """Load single-asset statistic modules."""
        loaded_count = 0
        
        # Categories for single-asset modules
        categories = ['basic', 'technical', 'statistical']
        
        for category in categories:
            category_path = modules_path / category
            if not category_path.exists():
                continue
            
            try:
                category_loaded = self._load_modules_from_directory(
                    category_path, category, BaseStatistic, self.single_asset_modules
                )
                loaded_count += category_loaded
                
                self.logger.debug(
                    f"Loaded {category_loaded} modules from {category} category"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to load modules from {category}: {e}")
        
        return loaded_count
    
    def _load_pair_trading_modules(self, modules_path: Path) -> int:
        """Load pair trading statistic modules."""
        loaded_count = 0
        
        pair_trading_path = modules_path / "pair_trading"
        if not pair_trading_path.exists():
            return 0
        
        # Subcategories for pair trading
        subcategories = ['correlation', 'cointegration', 'spread', 'hedge_ratio', 'momentum', 'volume']
        
        for subcategory in subcategories:
            subcategory_path = pair_trading_path / subcategory
            if not subcategory_path.exists():
                continue
            
            try:
                subcategory_loaded = self._load_modules_from_directory(
                    subcategory_path, f"pair_trading.{subcategory}",
                    BasePairStatistic, self.pair_trading_modules
                )
                loaded_count += subcategory_loaded
                
                self.logger.debug(
                    f"Loaded {subcategory_loaded} modules from "
                    f"pair_trading.{subcategory} subcategory"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to load modules from "
                    f"pair_trading.{subcategory}: {e}"
                )
        
        return loaded_count
    
    def _load_modules_from_directory(self,
                                     directory_path: Path,
                                     category: str,
                                     base_class: Type,
                                     module_dict: Dict[str, Type]) -> int:
        """Load all valid modules from a directory."""
        loaded_count = 0
        
        for py_file in directory_path.glob("*.py"):
            if (py_file.name.startswith("__") or
                    py_file.name.startswith("base_")):
                continue
            
            try:
                module_loaded = self._load_single_module(
                    py_file, category, base_class, module_dict
                )
                if module_loaded:
                    loaded_count += 1
                    
            except Exception as e:
                self.logger.warning(
                    f"Failed to load module {py_file.name}: {e}"
                )
        
        return loaded_count
    
    def _load_single_module(self,
                            module_file: Path,
                            category: str,
                            base_class: Type,
                            module_dict: Dict[str, Type]) -> bool:
        """Load a single module file and extract statistic classes."""
        module_name = module_file.stem
        
        # Build import path
        relative_path = module_file.relative_to(
            Path(__file__).parent / "modules"
        )
        import_path_parts = list(relative_path.parts[:-1]) + [module_name]
        import_path = (f"analytics.stats.modules."
                       f"{'.'.join(import_path_parts)}")
        
        try:
            # Import the module
            module = importlib.import_module(import_path)
            
            # Find classes that inherit from the base class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, base_class) and
                    obj != base_class and
                        not name.startswith('Base')):
                    
                    # Validate the class
                    if self._validate_statistic_class(obj, base_class):
                        full_name = f"{category}.{name}"
                        module_dict[full_name] = obj
                        
                        # Store metadata
                        self.module_metadata[full_name] = {
                            'class_name': name,
                            'module_path': import_path,
                            'category': category,
                            'file_path': str(module_file),
                            'base_class': base_class.__name__
                        }
                        
                        self.logger.debug(f"Loaded module: {full_name}")
                        return True
                        
        except Exception as e:
            self.logger.warning(f"Module import failed for {module_file}: {e}")
            return False
        
        return False
    
    def _validate_statistic_class(self, cls: Type, base_class: Type) -> bool:
        """Validate that a class meets requirements for statistics."""
        try:
            # Check for required method
            if hasattr(cls, 'calculate'):
                return True
            self.logger.debug(f"Class {cls.__name__} missing calculate method")
            return False
        except Exception as e:
            self.logger.debug(f"Validation failed for {cls.__name__}: {e}")
            return False
    
    def get_module_count(self) -> Dict[str, int]:
        """Get count of loaded modules by category."""
        return {
            'single_asset': len(self.single_asset_modules),            'pair_trading': len(self.pair_trading_modules)
        }
    
    def get_module_info(self) -> Dict[str, Dict[str, Type]]:
        """Get information about loaded modules."""
        return {
            'single_asset': self.single_asset_modules,
            'pair_trading': self.pair_trading_modules
        }
    
    def calculate_single_asset_statistics(self, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """Calculate all single asset statistics."""
        results = []
        
        for module_name, module_class in self.single_asset_modules.items():
            try:
                instance = module_class()
                result = instance.calculate(data)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error calculating {module_name} for {symbol}: {e}")
        
        return results
    
    def calculate_pair_statistics(self, data: pd.DataFrame, symbol1: str, symbol2: str) -> List[Dict]:
        """Calculate all pair trading statistics."""
        results = []
        
        for module_name, module_class in self.pair_trading_modules.items():
            try:
                instance = module_class()
                
                # Check if module uses new signature (data1, data2, **kwargs)
                import inspect
                sig = inspect.signature(instance.calculate)
                params = list(sig.parameters.keys())
                
                if len(params) >= 3 and 'data2' in params:
                    # New signature: split data into data1 and data2
                    data1_cols = [col for col in data.columns if col.startswith('asset_1_')]
                    data2_cols = [col for col in data.columns if col.startswith('asset_2_')]
                    
                    data1 = data[data1_cols].copy()
                    data1.columns = [col.replace('asset_1_', '') for col in data1.columns]
                    
                    data2 = data[data2_cols].copy()
                    data2.columns = [col.replace('asset_2_', '') for col in data2.columns]
                    
                    raw_result = instance.calculate(data1, data2, symbol1=symbol1, symbol2=symbol2)
                else:
                    # Old signature: pass full data
                    raw_result = instance.calculate(data)
                
                if raw_result and isinstance(raw_result, dict):
                    # Format the result for database storage
                    formatted_result = self._format_pair_result(raw_result, module_name, symbol1, symbol2)
                    if formatted_result:
                        results.append(formatted_result)
            except Exception as e:
                self.logger.error(f"Error calculating {module_name} for {symbol1}/{symbol2}: {e}")
        
        return results
        """Calculate all pair trading statistics."""
        results = []
        
        for module_name, module_class in self.pair_trading_modules.items():
            try:
                instance = module_class()
                
                # Check if module uses the new signature (data1, data2, **kwargs)
                if hasattr(instance, 'calculate') and hasattr(instance.calculate, '__code__'):
                    # Get function signature
                    arg_count = instance.calculate.__code__.co_argcount
                    arg_names = instance.calculate.__code__.co_varnames[:arg_count]
                    
                    if len(arg_names) >= 3 and 'data2' in arg_names:
                        # New signature: split data into data1 and data2
                        data1 = pd.DataFrame()
                        data2 = pd.DataFrame()
                        
                        # Extract asset 1 data
                        asset1_cols = [col for col in data.columns if col.startswith('asset_1_')]
                        if asset1_cols:
                            data1 = data[asset1_cols].copy()
                            # Remove asset_1_ prefix
                            data1.columns = [col.replace('asset_1_', '') for col in data1.columns]
                        
                        # Extract asset 2 data
                        asset2_cols = [col for col in data.columns if col.startswith('asset_2_')]
                        if asset2_cols:
                            data2 = data[asset2_cols].copy()
                            # Remove asset_2_ prefix  
                            data2.columns = [col.replace('asset_2_', '') for col in data2.columns]
                        
                        raw_result = instance.calculate(data1, data2, symbol1=symbol1, symbol2=symbol2)
                    else:
                        # Old signature: pass full data
                        raw_result = instance.calculate(data)
                else:
                    # Fallback: pass full data
                    raw_result = instance.calculate(data)
                
                if raw_result and isinstance(raw_result, dict):
                    # Format the result for database storage
                    formatted_result = self._format_pair_result(raw_result, module_name, symbol1, symbol2)
                    if formatted_result:
                        results.append(formatted_result)
            except Exception as e:
                self.logger.error(f"Error calculating {module_name} for {symbol1}/{symbol2}: {e}")
        
        return results
    
    def _format_pair_result(self, raw_result: Dict, module_name: str, symbol1: str, symbol2: str) -> Dict:
        """Format a raw module result for database storage."""
        # Extract the main statistic name and value
        statistic_name = None
        statistic_value = None
        metadata = {}
        
        # Look for the main statistic (usually the first non-metadata key)
        for key, value in raw_result.items():
            if key in ['data_points_used', 'metadata', 'symbol1', 'symbol2']:
                metadata[key] = value
            elif statistic_name is None:
                statistic_name = key
                statistic_value = value
            else:
                metadata[key] = value
        
        if statistic_name and statistic_value is not None:
            return {
                'statistic_name': statistic_name,
                'value': statistic_value,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'metadata': metadata,
                'module_name': module_name,
                'category': 'pair_trading'
            }
        
        return None
