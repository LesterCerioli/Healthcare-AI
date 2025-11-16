import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import json
import logging
import warnings
import time
from pathlib import Path
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor
import gc
from contextlib import contextmanager
import math
from scipy import stats as scipy_stats

"""
Enhanced Robust Statistical Data Analysis Framework with PyTorch
Now with comprehensive statistical calculations and hypothesis testing
"""


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataType(Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED_TEXT = "unstructured_text"
    UNSTRUCTURED_IMAGE = "unstructured_image"
    UNSTRUCTURED_AUDIO = "unstructured_audio"
    UNSTRUCTURED_NUMERICAL = "unstructured_numerical"

class StatisticalTestType(Enum):
    T_TEST_ONE_SAMPLE = "one_sample_t_test"
    T_TEST_INDEPENDENT = "independent_t_test"
    T_TEST_PAIRED = "paired_t_test"
    ANOVA_ONE_WAY = "one_way_anova"
    CHI_SQUARE = "chi_square"
    NORMALITY_TEST = "normality_test"
    CORRELATION_TEST = "correlation_test"

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataLoadingError(Exception):
    """Custom exception for data loading errors"""
    pass

class StatisticalCalculationError(Exception):
    """Custom exception for statistical calculation errors"""
    pass

@dataclass
class StatisticalSummary:
    """Comprehensive container for statistical summary results"""
    mean: torch.Tensor
    std: torch.Tensor
    median: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor
    quantiles: torch.Tensor
    shape: torch.Size
    data_type: str
    skewness: Optional[torch.Tensor] = None
    kurtosis: Optional[torch.Tensor] = None
    variance: Optional[torch.Tensor] = None
    confidence_intervals: Optional[Dict[str, torch.Tensor]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.tolist() if value.numel() > 1 else value.item()
            elif isinstance(value, torch.Size):
                result[key] = list(value)
        return result

@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results"""
    test_type: str
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    significance_level: float = 0.05
    rejected: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AdvancedStatistics:
    """Container for advanced statistical measures"""
    geometric_mean: torch.Tensor
    harmonic_mean: torch.Tensor
    mode: torch.Tensor
    range: torch.Tensor
    iqr: torch.Tensor
    mad: torch.Tensor  # Mean Absolute Deviation
    cv: torch.Tensor   # Coefficient of Variation
    z_scores: torch.Tensor
    outliers: torch.Tensor
    entropy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.tolist() if value.numel() > 1 else value.item()
        return result

@dataclass
class DataProfile:
    """Data profile and quality metrics"""
    total_samples: int
    total_features: int
    missing_values: int
    infinite_values: int
    memory_usage_mb: float
    load_time_seconds: float
    data_quality_score: float
    warnings: List[str]

class StatisticalCalculator:
    """Robust statistical calculations with comprehensive error handling"""
    
    @staticmethod
    def compute_advanced_statistics(data: torch.Tensor) -> AdvancedStatistics:
        """Compute advanced statistical measures"""
        with torch.no_grad():
            # Basic measures
            geometric_mean = StatisticalCalculator._geometric_mean(data)
            harmonic_mean = StatisticalCalculator._harmonic_mean(data)
            mode = StatisticalCalculator._mode(data)
            data_range = data.max() - data.min()
            iqr = StatisticalCalculator._interquartile_range(data)
            mad = StatisticalCalculator._mean_absolute_deviation(data)
            cv = StatisticalCalculator._coefficient_of_variation(data)
            z_scores = StatisticalCalculator._compute_z_scores(data)
            outliers = StatisticalCalculator._detect_outliers(data)
            entropy = StatisticalCalculator._compute_entropy(data)
            
            return AdvancedStatistics(
                geometric_mean=geometric_mean.cpu(),
                harmonic_mean=harmonic_mean.cpu(),
                mode=mode.cpu(),
                range=data_range.cpu(),
                iqr=iqr.cpu(),
                mad=mad.cpu(),
                cv=cv.cpu(),
                z_scores=z_scores.cpu(),
                outliers=outliers.cpu(),
                entropy=entropy
            )
    
    @staticmethod
    def _geometric_mean(data: torch.Tensor) -> torch.Tensor:
        """Compute geometric mean"""
        if (data <= 0).any():
            return torch.tensor(float('nan'))
        return torch.exp(torch.mean(torch.log(data)))
    
    @staticmethod
    def _harmonic_mean(data: torch.Tensor) -> torch.Tensor:
        """Compute harmonic mean"""
        if (data <= 0).any():
            return torch.tensor(float('nan'))
        return data.numel() / torch.sum(1.0 / data)
    
    @staticmethod
    def _mode(data: torch.Tensor) -> torch.Tensor:
        """Compute mode(s) of the data"""
        if data.dim() > 1:
            
            modes = []
            for i in range(data.shape[1]):
                values, counts = torch.unique(data[:, i], return_counts=True)
                modes.append(values[counts.argmax()])
            return torch.tensor(modes)
        else:
            values, counts = torch.unique(data, return_counts=True)
            return values[counts.argmax()]
    
    @staticmethod
    def _interquartile_range(data: torch.Tensor) -> torch.Tensor:
        """Compute interquartile range"""
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        return q3 - q1
    
    @staticmethod
    def _mean_absolute_deviation(data: torch.Tensor) -> torch.Tensor:
        """Compute mean absolute deviation"""
        return torch.mean(torch.abs(data - data.mean()))
    
    @staticmethod
    def _coefficient_of_variation(data: torch.Tensor) -> torch.Tensor:
        """Compute coefficient of variation"""
        mean = data.mean()
        if mean == 0:
            return torch.tensor(float('inf'))
        return data.std() / mean
    
    @staticmethod
    def _compute_z_scores(data: torch.Tensor) -> torch.Tensor:
        """Compute z-scores for standardization"""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return torch.zeros_like(data)
        return (data - mean) / std
    
    @staticmethod
    def _detect_outliers(data: torch.Tensor, method: str = 'iqr') -> torch.Tensor:
        """Detect outliers using specified method"""
        if method == 'iqr':
            q1 = torch.quantile(data, 0.25)
            q3 = torch.quantile(data, 0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = StatisticalCalculator._compute_z_scores(data)
            return torch.abs(z_scores) > 3
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def _compute_entropy(data: torch.Tensor, bins: int = 10) -> Optional[float]:
        """Compute Shannon entropy of the data distribution"""
        try:
            if data.numel() < 2:
                return None
            
            
            data_np = data.cpu().numpy()
            hist, _ = np.histogram(data_np, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero bins to avoid log(0)
            
            return float(-np.sum(hist * np.log(hist)) / np.log(2))  # Bits
        except:
            return None
    
    @staticmethod
    def perform_hypothesis_test(data: torch.Tensor, 
                              test_type: StatisticalTestType,
                              **kwargs) -> HypothesisTestResult:
        """Perform various statistical hypothesis tests"""
        try:
            if test_type == StatisticalTestType.T_TEST_ONE_SAMPLE:
                return StatisticalCalculator._one_sample_t_test(data, **kwargs)
            elif test_type == StatisticalTestType.T_TEST_INDEPENDENT:
                return StatisticalCalculator._independent_t_test(data, **kwargs)
            elif test_type == StatisticalTestType.NORMALITY_TEST:
                return StatisticalCalculator._normality_test(data, **kwargs)
            elif test_type == StatisticalTestType.CORRELATION_TEST:
                return StatisticalCalculator._correlation_test(data, **kwargs)
            else:
                raise StatisticalCalculationError(f"Unsupported test type: {test_type}")
        except Exception as e:
            raise StatisticalCalculationError(f"Hypothesis test failed: {str(e)}")
    
    @staticmethod
    def _one_sample_t_test(data: torch.Tensor, 
                          population_mean: float = 0.0) -> HypothesisTestResult:
        """One-sample t-test against a population mean"""
        if data.dim() > 1:
            raise ValueError("One-sample t-test requires 1D data")
        
        data_np = data.cpu().numpy()
        t_stat, p_value = scipy_stats.ttest_1samp(data_np, population_mean)
        
        
        effect_size = (data.mean().item() - population_mean) / data.std().item()
        
        
        n = len(data_np)
        sem = data.std().item() / math.sqrt(n)
        t_critical = scipy_stats.t.ppf(0.975, n - 1)
        ci_lower = data.mean().item() - t_critical * sem
        ci_upper = data.mean().item() + t_critical * sem
        
        return HypothesisTestResult(
            test_type=StatisticalTestType.T_TEST_ONE_SAMPLE.value,
            test_statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=n - 1,
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            rejected=p_value < 0.05
        )
    
    @staticmethod
    def _independent_t_test(data: torch.Tensor, 
                          group_indices: List[int]) -> HypothesisTestResult:
        """Independent samples t-test"""
        if data.dim() != 2 or data.shape[1] != len(group_indices):
            raise ValueError("Data shape must match group indices")
        
        groups = {}
        for i, group_id in enumerate(group_indices):
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(data[:, i].cpu().numpy())
        
        if len(groups) != 2:
            raise ValueError("Independent t-test requires exactly 2 groups")
        
        group1, group2 = list(groups.values())
        group1_flat = np.concatenate([g[~np.isnan(g)] for g in group1])
        group2_flat = np.concatenate([g[~np.isnan(g)] for g in group2])
        
        t_stat, p_value = scipy_stats.ttest_ind(group1_flat, group2_flat, equal_var=False)
        
        
        pooled_std = math.sqrt((len(group1_flat) - 1) * np.var(group1_flat) + 
                             (len(group2_flat) - 1) * np.var(group2_flat)) / \
                     (len(group1_flat) + len(group2_flat) - 2)
        effect_size = (np.mean(group1_flat) - np.mean(group2_flat)) / pooled_std
        
        return HypothesisTestResult(
            test_type=StatisticalTestType.T_TEST_INDEPENDENT.value,
            test_statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=len(group1_flat) + len(group2_flat) - 2,
            effect_size=float(effect_size),
            rejected=p_value < 0.05
        )
    
    @staticmethod
    def _normality_test(data: torch.Tensor, 
                       test_method: str = 'shapiro') -> HypothesisTestResult:
        """Test for normality using specified method"""
        if data.dim() > 1:
            # Flatten for normality test
            data_flat = data.flatten().cpu().numpy()
        else:
            data_flat = data.cpu().numpy()
        
        data_clean = data_flat[~np.isnan(data_flat)]
        
        if test_method == 'shapiro':
            if len(data_clean) < 3 or len(data_clean) > 5000:
                raise ValueError("Shapiro-Wilk test requires 3-5000 samples")
            test_stat, p_value = scipy_stats.shapiro(data_clean)
        elif test_method == 'normaltest':
            test_stat, p_value = scipy_stats.normaltest(data_clean)
        else:
            raise ValueError(f"Unknown normality test method: {test_method}")
        
        return HypothesisTestResult(
            test_type=f"normality_test_{test_method}",
            test_statistic=float(test_stat),
            p_value=float(p_value),
            rejected=p_value < 0.05
        )
    
    @staticmethod
    def _correlation_test(data: torch.Tensor, 
                        method: str = 'pearson') -> HypothesisTestResult:
        """Test correlation between variables"""
        if data.dim() != 2 or data.shape[1] != 2:
            raise ValueError("Correlation test requires 2D data with exactly 2 columns")
        
        x = data[:, 0].cpu().numpy()
        y = data[:, 1].cpu().numpy()
        
        
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if method == 'pearson':
            corr, p_value = scipy_stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, p_value = scipy_stats.spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, p_value = scipy_stats.kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return HypothesisTestResult(
            test_type=f"correlation_test_{method}",
            test_statistic=float(corr),
            p_value=float(p_value),
            rejected=p_value < 0.05
        )
    
    @staticmethod
    def compute_distribution_fit(data: torch.Tensor, 
                               distributions: List[str] = None) -> Dict[str, Any]:
        """Fit various distributions to data and return best fit"""
        if distributions is None:
            distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm']
        
        data_np = data.cpu().numpy()
        data_clean = data_np[~np.isnan(data_np)]
        
        results = {}
        for dist_name in distributions:
            try:
                if dist_name == 'norm':
                    params = scipy_stats.norm.fit(data_clean)
                    fitted_dist = scipy_stats.norm(*params)
                elif dist_name == 'expon':
                    params = scipy_stats.expon.fit(data_clean)
                    fitted_dist = scipy_stats.expon(*params)
                elif dist_name == 'gamma':
                    params = scipy_stats.gamma.fit(data_clean)
                    fitted_dist = scipy_stats.gamma(*params)
                elif dist_name == 'beta':
                    params = scipy_stats.beta.fit(data_clean)
                    fitted_dist = scipy_stats.beta(*params)
                elif dist_name == 'lognorm':
                    params = scipy_stats.lognorm.fit(data_clean)
                    fitted_dist = scipy_stats.lognorm(*params)
                else:
                    continue
                
                
                ks_stat, ks_pvalue = scipy_stats.kstest(data_clean, fitted_dist.cdf)
                
                
                log_likelihood = np.sum(fitted_dist.logpdf(data_clean))
                k = len(params)
                n = len(data_clean)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                results[dist_name] = {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic
                }
                
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name} distribution: {e}")
                continue
        
        
        if results:
            best_fit = min(results.items(), key=lambda x: x[1]['aic'])
            return {
                'best_fit': best_fit[0],
                'all_fits': results,
                'best_fit_metrics': best_fit[1]
            }
        else:
            return {'best_fit': None, 'all_fits': {}}

class DataValidator:
    """Robust data validation and cleaning"""
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor) -> Tuple[bool, List[str]]:
        """Validate tensor for statistical analysis"""
        warnings = []
        
        if tensor.numel() == 0:
            raise DataValidationError("Empty tensor provided")
        
        
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            warnings.append(f"Tensor contains {nan_count} NaN values")
        
        
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            warnings.append(f"Tensor contains {inf_count} infinite values")
        
        
        if not tensor.is_floating_point():
            warnings.append("Tensor converted to float32 for statistical operations")
            tensor = tensor.float()
        
        
        if tensor.std() == 0:
            warnings.append("Tensor has zero variance (constant values)")
        
        return len(warnings) == 0, warnings
    
    @staticmethod
    def clean_tensor(tensor: torch.Tensor, 
                    handle_nan: str = 'mean',
                    handle_inf: str = 'clip') -> torch.Tensor:
        """Clean tensor by handling NaN and infinite values"""
        tensor = tensor.clone().float()
        
        
        if torch.isnan(tensor).any():
            if handle_nan == 'mean':
                nan_mask = torch.isnan(tensor)
                tensor[nan_mask] = tensor[~nan_mask].mean()
            elif handle_nan == 'drop':
                tensor = tensor[~torch.isnan(tensor).any(dim=1)] if tensor.dim() > 1 else tensor[~torch.isnan(tensor)]
            elif handle_nan == 'zero':
                tensor[torch.isnan(tensor)] = 0
        
        
        if torch.isinf(tensor).any():
            if handle_inf == 'clip':
                finite_vals = tensor[~torch.isinf(tensor)]
                if finite_vals.numel() > 0:
                    max_val = finite_vals.max()
                    min_val = finite_vals.min()
                    tensor[torch.isinf(tensor) & (tensor > 0)] = max_val
                    tensor[torch.isinf(tensor) & (tensor < 0)] = min_val
            elif handle_inf == 'drop':
                tensor = tensor[~torch.isinf(tensor).any(dim=1)] if tensor.dim() > 1 else tensor[~torch.isinf(tensor)]
        
        return tensor

# ... (Keep all your existing DataSource, StructuredDataSource, UnstructuredDataSource classes exactly as they are) ...

class StatisticalAnalyzer:
    """Enhanced robust statistical analyzer with comprehensive statistical calculations"""
    
    def __init__(self, 
                 device: str = "auto",
                 confidence_level: float = 0.95,
                 enable_caching: bool = True):
        
        self.device = self._setup_device(device)
        self.confidence_level = confidence_level
        self.enable_caching = enable_caching
        
        self.data_source = None
        self.raw_data = None
        self._cache = {}
        
        logger.info(f"StatisticalAnalyzer initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup PyTorch device with validation"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                # Clear GPU memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                device = "cpu"
        
        try:
            dev = torch.device(device)
            # Test device
            if dev.type == "cuda":
                torch.zeros(1).to(dev)
            return dev
        except Exception as e:
            logger.warning(f"Device {device} unavailable, falling back to CPU: {e}")
            return torch.device("cpu")
    
    @contextmanager
    def _timer(self, operation: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{operation} completed in {elapsed:.4f}s")
    
    def load_data(self, data_source: DataSource, use_cache: bool = True) -> None:
        """Load data from source with caching support"""
        cache_key = f"data_{data_source.source_id}"
        
        if use_cache and self.enable_caching and cache_key in self._cache:
            logger.info(f"Loading data from cache: {cache_key}")
            self.raw_data = self._cache[cache_key]
            self.data_source = data_source
            return
        
        with self._timer("Data loading"):
            self.raw_data = data_source.validate_and_load().to(self.device)
            self.data_source = data_source
            
            if self.enable_caching:
                self._cache[cache_key] = self.raw_data.cpu()  # Cache on CPU
    
    def compute_comprehensive_statistics(self) -> StatisticalSummary:
        """Compute comprehensive statistical summary"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        with self._timer("Statistics computation"):
            data = self.raw_data
            
            # Handle empty data
            if data.numel() == 0:
                logger.warning("Empty data tensor detected")
                return self._create_empty_summary(data)
            
            with torch.no_grad():
                # Basic statistics
                mean = torch.mean(data)
                std = torch.std(data)
                median = torch.median(data)
                min_val = torch.min(data)
                max_val = torch.max(data)
                
                # Quantiles
                quantiles = torch.quantile(
                    data, 
                    torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=self.device)
                )
                
                # Advanced statistics
                variance = torch.var(data)
                skewness = self._compute_skewness(data)
                kurtosis = self._compute_kurtosis(data)
                confidence_intervals = self._compute_confidence_intervals(data, mean, std)
                
                return StatisticalSummary(
                    mean=mean.cpu(),
                    std=std.cpu(),
                    median=median.cpu(),
                    min=min_val.cpu(),
                    max=max_val.cpu(),
                    quantiles=quantiles.cpu(),
                    shape=data.shape,
                    data_type=str(data.dtype),
                    skewness=skewness.cpu(),
                    kurtosis=kurtosis.cpu(),
                    variance=variance.cpu(),
                    confidence_intervals=confidence_intervals
                )
    
    def compute_advanced_statistics(self) -> AdvancedStatistics:
        """Compute advanced statistical measures"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        with self._timer("Advanced statistics computation"):
            return StatisticalCalculator.compute_advanced_statistics(self.raw_data)
    
    def perform_hypothesis_test(self, 
                              test_type: StatisticalTestType,
                              **kwargs) -> HypothesisTestResult:
        """Perform statistical hypothesis test on loaded data"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        with self._timer(f"Hypothesis test: {test_type.value}"):
            return StatisticalCalculator.perform_hypothesis_test(
                self.raw_data, test_type, **kwargs
            )
    
    def compute_distribution_fit(self, 
                               distributions: List[str] = None) -> Dict[str, Any]:
        """Fit probability distributions to data"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        with self._timer("Distribution fitting"):
            return StatisticalCalculator.compute_distribution_fit(
                self.raw_data, distributions
            )
    
    def compute_time_series_analysis(self, 
                                   window_size: int = 5) -> Dict[str, Any]:
        """Compute time series analysis metrics if data is sequential"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        data = self.raw_data
        if data.dim() != 1:
            logger.warning("Time series analysis typically requires 1D data")
        
        with torch.no_grad():
            # Simple time series metrics
            differences = data[1:] - data[:-1]
            autocorr_lag1 = torch.corrcoef(
                torch.stack([data[:-1], data[1:]])
            )[0, 1].item() if len(data) > 1 else float('nan')
            
            # Rolling statistics
            if len(data) >= window_size:
                rolling_mean = torch.conv1d(
                    data.unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, window_size) / window_size,
                    padding=window_size//2
                ).squeeze()
                
                rolling_std = torch.sqrt(torch.conv1d(
                    (data - rolling_mean).square().unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, window_size) / window_size,
                    padding=window_size//2
                ).squeeze())
            else:
                rolling_mean = torch.tensor([float('nan')])
                rolling_std = torch.tensor([float('nan')])
            
            return {
                'autocorrelation_lag1': autocorr_lag1,
                'mean_absolute_change': torch.mean(torch.abs(differences)).item(),
                'stationarity_p_value': self._check_stationarity(data),
                'rolling_mean': rolling_mean.cpu().tolist(),
                'rolling_std': rolling_std.cpu().tolist()
            }
    
    def _check_stationarity(self, data: torch.Tensor) -> float:
        """Check stationarity using Augmented Dickey-Fuller test (simplified)"""
        try:
            from statsmodels.tsa.stattools import adfuller
            data_np = data.cpu().numpy()
            result = adfuller(data_np)
            return float(result[1])  # p-value
        except ImportError:
            logger.warning("statsmodels not available for stationarity test")
            return float('nan')
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return float('nan')
    
    def _compute_skewness(self, data: torch.Tensor) -> torch.Tensor:
        """Compute skewness (third standardized moment)"""
        if data.numel() < 3:
            return torch.tensor(float('nan'))
        
        mean = data.mean()
        std = data.std()
        if std == 0:
            return torch.tensor(0.0)
        
        z_scores = (data - mean) / std
        return torch.mean(z_scores ** 3)
    
    def _compute_kurtosis(self, data: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis (fourth standardized moment)"""
        if data.numel() < 4:
            return torch.tensor(float('nan'))
        
        mean = data.mean()
        std = data.std()
        if std == 0:
            return torch.tensor(0.0)
        
        z_scores = (data - mean) / std
        return torch.mean(z_scores ** 4) - 3  # Excess kurtosis
    
    def _compute_confidence_intervals(self, 
                                    data: torch.Tensor, 
                                    mean: torch.Tensor, 
                                    std: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute confidence intervals for the mean"""
        if data.numel() < 2:
            return {}
        
        from scipy import stats
        try:
            # Using t-distribution for small samples, normal for large samples
            n = data.numel()
            if n <= 30:
                critical_value = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
            else:
                critical_value = stats.norm.ppf((1 + self.confidence_level) / 2)
            
            margin_of_error = critical_value * std / torch.sqrt(torch.tensor(n, dtype=torch.float32))
            
            return {
                "lower": (mean - margin_of_error).cpu(),
                "upper": (mean + margin_of_error).cpu(),
                "confidence_level": self.confidence_level
            }
        except ImportError:
            logger.warning("SciPy not available, skipping confidence intervals")
            return {}
    
    def _create_empty_summary(self, data: torch.Tensor) -> StatisticalSummary:
        """Create empty summary for empty data"""
        nan_tensor = torch.tensor(float('nan'))
        return StatisticalSummary(
            mean=nan_tensor,
            std=nan_tensor,
            median=nan_tensor,
            min=nan_tensor,
            max=nan_tensor,
            quantiles=torch.tensor([float('nan')] * 5),
            shape=data.shape,
            data_type="empty"
        )
    
    def compute_correlation_analysis(self) -> Dict[str, Any]:
        """Compute comprehensive correlation analysis"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        data = self.raw_data
        
        if data.dim() == 1:
            return {"warning": "Correlation requires multi-dimensional data"}
        
        with torch.no_grad():
            # Pearson correlation
            data_std = (data - data.mean(dim=0)) / data.std(dim=0)
            pearson_corr = torch.mm(data_std.T, data_std) / (data_std.shape[0] - 1)
            
            # Spearman correlation (rank-based)
            ranks = torch.argsort(torch.argsort(data, dim=0), dim=0).float()
            ranks_std = (ranks - ranks.mean(dim=0)) / ranks.std(dim=0)
            spearman_corr = torch.mm(ranks_std.T, ranks_std) / (ranks_std.shape[0] - 1)
            
            return {
                "pearson_correlation": pearson_corr.cpu().tolist(),
                "spearman_correlation": spearman_corr.cpu().tolist(),
                "correlation_strength": self._classify_correlation_strength(pearson_corr)
            }
    
    def _classify_correlation_strength(self, corr_matrix: torch.Tensor) -> List[List[str]]:
        """Classify correlation strength"""
        strength_map = [
            (0.9, 1.0, "very_strong"),
            (0.7, 0.9, "strong"),
            (0.5, 0.7, "moderate"),
            (0.3, 0.5, "weak"),
            (0.0, 0.3, "very_weak")
        ]
        
        classifications = []
        for i in range(corr_matrix.shape[0]):
            row_classifications = []
            for j in range(corr_matrix.shape[1]):
                if i == j:
                    row_classifications.append("self")
                else:
                    corr_val = abs(corr_matrix[i, j].item())
                    for min_val, max_val, strength in strength_map:
                        if min_val <= corr_val < max_val:
                            row_classifications.append(strength)
                            break
                    else:
                        row_classifications.append("very_weak")
            classifications.append(row_classifications)
        
        return classifications
    
    def generate_data_profile(self) -> DataProfile:
        """Generate comprehensive data profile"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        data = self.raw_data
        
        # Compute data quality metrics
        missing_values = torch.isnan(data).sum().item()
        infinite_values = torch.isinf(data).sum().item()
        
        # Memory usage
        memory_usage = data.element_size() * data.nelement() / 1024**2
        
        # Data quality score (0-1)
        total_elements = data.numel()
        quality_score = 1.0 - (missing_values + infinite_values) / total_elements if total_elements > 0 else 0
        
        warnings = []
        if missing_values > 0:
            warnings.append(f"Found {missing_values} missing values")
        if infinite_values > 0:
            warnings.append(f"Found {infinite_values} infinite values")
        if data.std() == 0:
            warnings.append("Data has zero variance")
        
        return DataProfile(
            total_samples=data.shape[0] if data.dim() > 0 else 1,
            total_features=data.shape[1] if data.dim() > 1 else 1,
            missing_values=missing_values,
            infinite_values=infinite_values,
            memory_usage_mb=memory_usage,
            load_time_seconds=self.data_source.load_time or 0,
            data_quality_score=quality_score,
            warnings=warnings
        )
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical report with all new statistical calculations"""
        if self.data_source is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Generating comprehensive statistical report")
        
        with self._timer("Report generation"):
            stats = self.compute_comprehensive_statistics()
            advanced_stats = self.compute_advanced_statistics()
            profile = self.generate_data_profile()
            metadata = self.data_source.get_metadata()
            
            report = {
                "metadata": metadata,
                "data_profile": asdict(profile),
                "basic_statistics": stats.to_dict(),
                "advanced_statistics": advanced_stats.to_dict(),
                "timestamp": time.time(),
                "analyzer_info": {
                    "device": str(self.device),
                    "pytorch_version": torch.__version__,
                    "confidence_level": self.confidence_level
                }
            }
            
            # Add correlation analysis for multi-dimensional data
            if self.raw_data.dim() > 1 and self.raw_data.shape[1] > 1:
                report["correlation_analysis"] = self.compute_correlation_analysis()
            
            # Add distribution fitting results
            try:
                report["distribution_fitting"] = self.compute_distribution_fit()
            except Exception as e:
                logger.warning(f"Distribution fitting failed: {e}")
                report["distribution_fitting"] = {"error": str(e)}
            
            # Add time series analysis if applicable
            if self.raw_data.dim() == 1 and len(self.raw_data) > 10:
                try:
                    report["time_series_analysis"] = self.compute_time_series_analysis()
                except Exception as e:
                    logger.warning(f"Time series analysis failed: {e}")
                    report["time_series_analysis"] = {"error": str(e)}
            
            # Add hypothesis test examples
            try:
                report["hypothesis_tests"] = {
                    "normality_test": self.perform_hypothesis_test(
                        StatisticalTestType.NORMALITY_TEST
                    ).to_dict(),
                    "one_sample_t_test": self.perform_hypothesis_test(
                        StatisticalTestType.T_TEST_ONE_SAMPLE, population_mean=0.0
                    ).to_dict()
                }
            except Exception as e:
                logger.warning(f"Hypothesis tests failed: {e}")
                report["hypothesis_tests"] = {"error": str(e)}
            
            return report
    
    def clear_cache(self) -> None:
        """Clear internal cache"""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cache cleared")

# Enhanced demonstration function
def demonstrate_enhanced_analyzer():
    """Demonstrate the enhanced statistical analyzer with robust statistical calculations"""
    
    print("=== Enhanced Statistical Analyzer Demonstration ===\n")
    
    # Initialize analyzer with caching
    analyzer = StatisticalAnalyzer(enable_caching=True)
    
    # Example 1: Comprehensive structured data analysis
    print("1. Comprehensive Structured Data Analysis")
    try:
        # Create sample structured data with realistic characteristics
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'normal_feature': np.random.normal(50, 15, n_samples),
            'exponential_feature': np.random.exponential(2, n_samples),
            'uniform_feature': np.random.uniform(0, 100, n_samples),
            'bimodal_feature': np.concatenate([
                np.random.normal(30, 5, n_samples//2),
                np.random.normal(70, 8, n_samples//2)
            ]),
            'category': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Add some realistic data issues
        sample_data.loc[10:15, 'normal_feature'] = np.nan
        sample_data.loc[20, 'exponential_feature'] = np.inf
        
        structured_source = StructuredDataSource(
            sample_data,
            numeric_columns=['normal_feature', 'exponential_feature', 
                           'uniform_feature', 'bimodal_feature'],
            source_id="enhanced_structured_demo"
        )
        
        analyzer.load_data(structured_source)
        report = analyzer.generate_comprehensive_report()
        
        print(f"   Data Shape: {report['data_profile']['total_samples']} samples, "
              f"{report['data_profile']['total_features']} features")
        print(f"   Data Quality Score: {report['data_profile']['data_quality_score']:.3f}")
        
        # Show advanced statistics
        adv_stats = report['advanced_statistics']
        print(f"   Geometric Mean: {adv_stats['geometric_mean']:.2f}")
        print(f"   Harmonic Mean: {adv_stats['harmonic_mean']:.2f}")
        print(f"   IQR: {adv_stats['iqr']:.2f}")
        print(f"   Outliers Detected: {sum(adv_stats['outliers'])}")
        
        # Show hypothesis test results
        norm_test = report['hypothesis_tests']['normality_test']
        print(f"   Normality Test p-value: {norm_test['p_value']:.4f}")
        print(f"   Data is Normal: {not norm_test['rejected']}")
        
    except Exception as e:
        print(f"   Enhanced structured data analysis failed: {e}")
    
    # Example 2: Advanced time series analysis
    print("\n2. Time Series Analysis")
    try:
        # Create synthetic time series data
        t = np.linspace(0, 4*np.pi, 500)
        time_series_data = np.sin(t) + 0.5 * np.random.normal(0, 0.2, 500) + 0.1 * t
        
        ts_source = UnstructuredDataSource(
            time_series_data,
            data_type=DataType.UNSTRUCTURED_NUMERICAL,
            source_id="time_series_demo"
        )
        
        analyzer.load_data(ts_source)
        report = analyzer.generate_comprehensive_report()
        
        if 'time_series_analysis' in report:
            ts_analysis = report['time_series_analysis']
            print(f"   Autocorrelation (lag1): {ts_analysis['autocorrelation_lag1']:.3f}")
            print(f"   Mean Absolute Change: {ts_analysis['mean_absolute_change']:.3f}")
            if 'stationarity_p_value' in ts_analysis:
                print(f"   Stationarity p-value: {ts_analysis['stationarity_p_value']:.4f}")
        
    except Exception as e:
        print(f"   Time series analysis failed: {e}")
    
    # Example 3: Distribution fitting demonstration
    print("\n3. Distribution Fitting")
    try:
        # Create data from specific distribution
        gamma_data = np.random.gamma(2, 2, 1000)
        
        dist_source = UnstructuredDataSource(
            gamma_data,
            data_type=DataType.UNSTRUCTURED_NUMERICAL,
            source_id="distribution_demo"
        )
        
        analyzer.load_data(dist_source)
        dist_fit = analyzer.compute_distribution_fit(['norm', 'gamma', 'expon'])
        
        if dist_fit['best_fit']:
            print(f"   Best fitting distribution: {dist_fit['best_fit']}")
            print(f"   Best fit AIC: {dist_fit['best_fit_metrics']['aic']:.2f}")
            
            # Show all fits
            for dist_name, metrics in dist_fit['all_fits'].items():
                print(f"   {dist_name}: AIC={metrics['aic']:.2f}, "
                      f"KS p-value={metrics['ks_pvalue']:.4f}")
        
    except Exception as e:
        print(f"   Distribution fitting failed: {e}")
    
    # Example 4: Multiple hypothesis tests
    print("\n4. Multiple Hypothesis Tests")
    try:
        # Test data against different null hypotheses
        test_data = np.random.normal(5, 2, 200)
        
        test_source = UnstructuredDataSource(
            test_data,
            data_type=DataType.UNSTRUCTURED_NUMERICAL,
            source_id="hypothesis_test_demo"
        )
        
        analyzer.load_data(test_source)
        
        # Perform various tests
        normality_result = analyzer.perform_hypothesis_test(
            StatisticalTestType.NORMALITY_TEST
        )
        
        t_test_result = analyzer.perform_hypothesis_test(
            StatisticalTestType.T_TEST_ONE_SAMPLE, population_mean=0.0
        )
        
        print(f"   Normality test p-value: {normality_result.p_value:.4f}")
        print(f"   One-sample t-test p-value: {t_test_result.p_value:.4f}")
        print(f"   Effect size (Cohen's d): {t_test_result.effect_size:.3f}")
        print(f"   Significant difference from 0: {t_test_result.rejected}")
        
    except Exception as e:
        print(f"   Hypothesis testing failed: {e}")
    
    # Clear cache
    analyzer.clear_cache()
    print("\n=== Enhanced Demonstration Completed ===")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    demonstrate_enhanced_analyzer()