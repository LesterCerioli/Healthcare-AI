#!/usr/bin/env python3
"""
Diabetes Glucose Pattern Analyzer
Analyzes temporal glucose patterns with focus on nocturnal hypoglycemia
and daytime hyperglycemia in Type 1 diabetic patients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Clinical Configuration
# -----------------------------
@dataclass
class GlucoseConfig:
    """Clinical configuration for glucose analysis"""
    
    
    SEVERE_HYPO: int = 54        # Severe hypoglycemia
    MILD_HYPO: int = 70          # Mild hypoglycemia
    NORMAL_MIN: int = 70         # Normal minimum
    NORMAL_MAX: int = 120        # Normal maximum
    MILD_HYPER: int = 180        # Mild hyperglycemia
    SEVERE_HYPER: int = 250      # Severe hyperglycemia
    CRITICAL_HYPER: int = 300    # Critical hyperglycemia
    
    
    TIME_PERIODS: Dict[str, Tuple[time, time]] = None
    
    
    RISK_CATEGORIES: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.TIME_PERIODS is None:
            self.TIME_PERIODS = {
                'overnight': (time(0, 0), time(5, 59)),
                'morning': (time(6, 0), time(11, 59)),
                'afternoon': (time(12, 0), time(17, 59)),
                'evening': (time(18, 0), time(23, 59))
            }
        
        if self.RISK_CATEGORIES is None:
            self.RISK_CATEGORIES = {
                'severe_hypoglycemia': (0, self.SEVERE_HYPO),
                'mild_hypoglycemia': (self.SEVERE_HYPO + 1, self.MILD_HYPO),
                'normal': (self.MILD_HYPO + 1, self.NORMAL_MAX),
                'mild_hyperglycemia': (self.NORMAL_MAX + 1, self.MILD_HYPER),
                'severe_hyperglycemia': (self.MILD_HYPER + 1, self.SEVERE_HYPER),
                'critical_hyperglycemia': (self.SEVERE_HYPER + 1, 600)
            }

CFG = GlucoseConfig()

# -----------------------------
# Realistic Data Generator
# -----------------------------
class GlucoseDataGenerator:
    """Generates realistic glucose monitoring data"""
    
    def __init__(self, config: GlucoseConfig):
        self.config = config
        
    def generate_patient_data(self, patient_id: str, days: int = 30) -> pd.DataFrame:
        """Generates glucose data for a Type 1 diabetic patient"""
        
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            
            daily_pattern = self._generate_daily_pattern(current_date)
            data.extend(daily_pattern)
        
        df = pd.DataFrame(data)
        df['patient_id'] = patient_id
        return df
    
    def _generate_daily_pattern(self, date: datetime) -> List[Dict]:
        """Generates daily glucose pattern with realistic scenarios"""
        measurements = []
        
        
        time_slots = [
            (time(2, 0), "overnight"),    # Overnight - hypoglycemia risk
            (time(6, 30), "morning"),     # Fasting
            (time(8, 0), "morning"),      # Post-breakfast
            (time(12, 0), "afternoon"),   # Pre-lunch
            (time(14, 0), "afternoon"),   # Post-lunch
            (time(18, 0), "evening"),     # Pre-dinner
            (time(20, 0), "evening"),     # Post-dinner
            (time(23, 0), "evening")      # Bedtime
        ]
        
        
        base_glucose = np.random.normal(180, 40)
        
        for time_slot, period in time_slots:
            glucose = self._calculate_glucose_value(base_glucose, period, time_slot)
            
            
            if period == 'overnight' and np.random.random() < 0.3:  # 30% chance
                glucose = np.random.randint(45, 65)
            
            
            if time_slot.hour in [8, 14, 20] and np.random.random() < 0.4:
                glucose += np.random.randint(50, 100)
            
            measurements.append({
                'timestamp': datetime.combine(date, time_slot),
                'glucose_mgdl': max(40, min(450, glucose)),  # Realistic limits
                'time_period': period,
                'time_of_day': time_slot.strftime('%H:%M'),
                'day_of_week': date.strftime('%A'),
                'is_weekend': date.weekday() >= 5
            })
        
        return measurements
    
    def _calculate_glucose_value(self, base: float, period: str, time_slot: time) -> float:
        """Calculates glucose value based on period and time"""
        
        
        period_factors = {
            'overnight': 0.7,   # Tendency for lower glucose
            'morning': 1.1,     # Dawn phenomenon
            'afternoon': 1.0,   # Stable
            'evening': 0.9      # Pre-sleep preparation
        }
        
       
        time_variation = np.random.normal(0, 25)
        adjusted_glucose = base * period_factors[period] + time_variation
        
        return adjusted_glucose

# -----------------------------
# Pattern Analyzer
# -----------------------------
class GlucosePatternAnalyzer:
    """Analyzes temporal glucose patterns"""
    
    def __init__(self, config: GlucoseConfig):
        self.config = config
        
    def analyze_glucose_data(self, df: pd.DataFrame) -> Dict:
        """Performs comprehensive glucose data analysis"""
        
        logger.info("🔍 Starting glucose pattern analysis...")
        
        analysis = {
            'summary_stats': self._calculate_summary_stats(df),
            'time_period_analysis': self._analyze_by_time_period(df),
            'hypoglycemia_analysis': self._analyze_hypoglycemia_patterns(df),
            'hyperglycemia_analysis': self._analyze_hyperglycemia_patterns(df),
            'risk_assessment': self._assess_risks(df),
            'temporal_patterns': self._analyze_temporal_patterns(df)
        }
        
        return analysis
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculates summary statistics"""
        
        glucose = df['glucose_mgdl']
        
        return {
            'total_measurements': len(df),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'glucose_stats': {
                'mean': glucose.mean(),
                'median': glucose.median(),
                'std': glucose.std(),
                'min': glucose.min(),
                'max': glucose.max()
            },
            'time_in_ranges': self._calculate_time_in_ranges(df),
            'variability_metrics': self._calculate_variability_metrics(df)
        }
    
    def _calculate_time_in_ranges(self, df: pd.DataFrame) -> Dict:
        """Calculates time in different glucose ranges"""
        
        total = len(df)
        ranges = {}
        
        for range_name, (low, high) in self.config.RISK_CATEGORIES.items():
            count = ((df['glucose_mgdl'] >= low) & (df['glucose_mgdl'] <= high)).sum()
            percentage = (count / total) * 100
            ranges[range_name] = {
                'count': count,
                'percentage': percentage,
                'range': f"{low}-{high} mg/dL"
            }
        
        return ranges
    
    def _calculate_variability_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculates glucose variability metrics"""
        
        glucose = df['glucose_mgdl']
        
        return {
            'cv_percentage': (glucose.std() / glucose.mean()) * 100,  # Coefficient of variation
            'mean_amplitude_glycemic_excursion': self._calculate_mage(glucose),
            'low_blood_glucose_index': self._calculate_lbgi(glucose),
            'high_blood_glucose_index': self._calculate_hbgi(glucose)
        }
    
    def _calculate_mage(self, glucose_series: pd.Series) -> float:
        """Calculates Mean Amplitude of Glycemic Excursion"""
        differences = glucose_series.diff().abs()
        return differences.mean()
    
    def _calculate_lbgi(self, glucose_series: pd.Series) -> float:
        """Calculates Low Blood Glucose Index"""
        low_glucose = glucose_series[glucose_series < self.config.MILD_HYPO]
        if len(low_glucose) == 0:
            return 0
        return ((self.config.MILD_HYPO - low_glucose) ** 2).mean() / 100
    
    def _calculate_hbgi(self, glucose_series: pd.Series) -> float:
        """Calculates High Blood Glucose Index"""
        high_glucose = glucose_series[glucose_series > self.config.MILD_HYPER]
        if len(high_glucose) == 0:
            return 0
        return ((high_glucose - self.config.MILD_HYPER) ** 2).mean() / 100
    
    def _analyze_by_time_period(self, df: pd.DataFrame) -> Dict:
        """Analyzes glucose by time period"""
        
        period_analysis = {}
        
        for period_name in self.config.TIME_PERIODS.keys():
            period_data = df[df['time_period'] == period_name]
            
            if len(period_data) > 0:
                glucose = period_data['glucose_mgdl']
                
                period_analysis[period_name] = {
                    'measurement_count': len(period_data),
                    'mean_glucose': glucose.mean(),
                    'hypoglycemia_events': (glucose < self.config.MILD_HYPO).sum(),
                    'hyperglycemia_events': (glucose > self.config.MILD_HYPER).sum(),
                    'time_in_normal_range': ((glucose >= self.config.NORMAL_MIN) & 
                                           (glucose <= self.config.NORMAL_MAX)).sum() / len(period_data) * 100
                }
        
        return period_analysis
    
    def _analyze_hypoglycemia_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyzes specific hypoglycemia patterns"""
        
        hypoglycemia_events = df[df['glucose_mgdl'] < self.config.MILD_HYPO]
        
        if len(hypoglycemia_events) == 0:
            return {'total_events': 0, 'analysis': 'No hypoglycemia events detected'}
        
        analysis = {
            'total_hypoglycemia_events': len(hypoglycemia_events),
            'severe_hypoglycemia_events': (hypoglycemia_events['glucose_mgdl'] < self.config.SEVERE_HYPO).sum(),
            'nocturnal_hypoglycemia': self._analyze_nocturnal_hypoglycemia(hypoglycemia_events),
            'temporal_distribution': self._analyze_hypoglycemia_timing(hypoglycemia_events),
            'recurrence_patterns': self._analyze_hypoglycemia_recurrence(hypoglycemia_events)
        }
        
        return analysis
    
    def _analyze_nocturnal_hypoglycemia(self, hypo_events: pd.DataFrame) -> Dict:
        """Analyzes nocturnal/overnight hypoglycemia"""
        
        nocturnal_periods = ['overnight', 'evening']
        nocturnal_hypo = hypo_events[hypo_events['time_period'].isin(nocturnal_periods)]
        
        return {
            'events_count': len(nocturnal_hypo),
            'percentage_of_total_hypo': (len(nocturnal_hypo) / len(hypo_events)) * 100,
            'average_glucose_nocturnal': nocturnal_hypo['glucose_mgdl'].mean() if len(nocturnal_hypo) > 0 else 0,
            'most_common_time': nocturnal_hypo['time_of_day'].mode().iloc[0] if len(nocturnal_hypo) > 0 else 'N/A'
        }
    
    def _analyze_hypoglycemia_timing(self, hypo_events: pd.DataFrame) -> Dict:
        """Analyzes hypoglycemia timing distribution"""
        
        timing_analysis = {}
        
        for period_name in self.config.TIME_PERIODS.keys():
            period_hypo = hypo_events[hypo_events['time_period'] == period_name]
            timing_analysis[period_name] = {
                'events': len(period_hypo),
                'percentage': (len(period_hypo) / len(hypo_events)) * 100
            }
        
        return timing_analysis
    
    def _analyze_hypoglycemia_recurrence(self, hypo_events: pd.DataFrame) -> Dict:
        """Analyzes hypoglycemia recurrence patterns"""
        
        
        hypo_events['date'] = hypo_events['timestamp'].dt.date
        daily_counts = hypo_events.groupby('date').size()
        
        return {
            'days_with_hypoglycemia': len(daily_counts),
            'max_events_per_day': daily_counts.max() if len(daily_counts) > 0 else 0,
            'average_events_per_day_with_hypo': daily_counts.mean() if len(daily_counts) > 0 else 0,
            'consecutive_days_with_hypo': self._find_consecutive_hypo_days(hypo_events)
        }
    
    def _find_consecutive_hypo_days(self, hypo_events: pd.DataFrame) -> int:
        """Finds longest sequence of consecutive days with hypoglycemia"""
        unique_dates = sorted(hypo_events['date'].unique())
        
        if not unique_dates:
            return 0
            
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(unique_dates)):
            if (unique_dates[i] - unique_dates[i-1]).days == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive
    
    def _analyze_hyperglycemia_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyzes hyperglycemia patterns"""
        
        hyper_events = df[df['glucose_mgdl'] > self.config.MILD_HYPER]
        critical_hyper = df[df['glucose_mgdl'] > self.config.CRITICAL_HYPER]
        
        analysis = {
            'total_hyperglycemia_events': len(hyper_events),
            'critical_hyperglycemia_events': len(critical_hyper),
            'percentage_time_above_300': (len(critical_hyper) / len(df)) * 100,
            'postprandial_hyperglycemia': self._analyze_postprandial_hyperglycemia(df),
            'dawn_phenomenon': self._analyze_dawn_phenomenon(df)
        }
        
        return analysis
    
    def _analyze_postprandial_hyperglycemia(self, df: pd.DataFrame) -> Dict:
        """Analyzes postprandial hyperglycemia (after meals)"""
        
        
        post_meal_times = ['08:00', '14:00', '20:00']
        post_meal_data = df[df['time_of_day'].isin(post_meal_times)]
        
        post_meal_hyper = post_meal_data[post_meal_data['glucose_mgdl'] > self.config.MILD_HYPER]
        
        return {
            'post_meal_measurements': len(post_meal_data),
            'post_meal_hyperglycemia_events': len(post_meal_hyper),
            'post_meal_hyper_percentage': (len(post_meal_hyper) / len(post_meal_data)) * 100,
            'average_post_meal_glucose': post_meal_data['glucose_mgdl'].mean()
        }
    
    def _analyze_dawn_phenomenon(self, df: pd.DataFrame) -> Dict:
        """Analyzes dawn phenomenon (morning glucose rise)"""
        
        morning_data = df[df['time_period'] == 'morning']
        nocturnal_data = df[df['time_period'] == 'overnight']
        
        if len(morning_data) > 0 and len(nocturnal_data) > 0:
            dawn_effect = morning_data['glucose_mgdl'].mean() - nocturnal_data['glucose_mgdl'].mean()
        else:
            dawn_effect = 0
        
        return {
            'dawn_phenomenon_present': dawn_effect > 20,  # Increase > 20 mg/dL
            'average_dawn_increase': dawn_effect,
            'morning_hyperglycemia_events': (morning_data['glucose_mgdl'] > self.config.MILD_HYPER).sum()
        }
    
    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Assesses clinical risks based on patterns"""
        
        risks = {
            'immediate_risks': [],
            'long_term_risks': [],
            'recommendations': []
        }
        
        
        nocturnal_hypo = df[(df['time_period'].isin(['overnight', 'evening'])) & 
                           (df['glucose_mgdl'] < self.config.MILD_HYPO)]
        
        if len(nocturnal_hypo) > 0:
            risks['immediate_risks'].append({
                'risk': 'Nocturnal Hypoglycemia',
                'severity': 'HIGH' if len(nocturnal_hypo) > 3 else 'MODERATE',
                'description': f'{len(nocturnal_hypo)} hypoglycemia events detected during night/evening',
                'action': 'Adjust basal insulin dose and consider bedtime snack'
            })
        
        
        critical_hyper = df[df['glucose_mgdl'] > self.config.CRITICAL_HYPER]
        if len(critical_hyper) > 0:
            risks['immediate_risks'].append({
                'risk': 'Critical Hyperglycemia',
                'severity': 'HIGH',
                'description': f'{len(critical_hyper)} measurements above 300 mg/dL - risk of ketoacidosis',
                'action': 'Immediately review insulin regimen'
            })
        
        
        cv = (df['glucose_mgdl'].std() / df['glucose_mgdl'].mean()) * 100
        if cv > 36:  # Coefficient of variation > 36% indicates high variability
            risks['long_term_risks'].append({
                'risk': 'High Glucose Variability',
                'description': f'CV = {cv:.1f}% - Increases risk of microvascular complications',
                'action': 'Optimize insulin regimen to reduce fluctuations'
            })
        
        
        time_in_range = self._calculate_time_in_ranges(df)
        normal_time = time_in_range['normal']['percentage']
        
        if normal_time < 70:
            risks['recommendations'].append(
                f"Increase time in normal range (currently {normal_time:.1f}%)"
            )
        
        return risks
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyzes advanced temporal patterns"""
        
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        
        hourly_patterns = df.groupby('hour')['glucose_mgdl'].agg(['mean', 'std', 'count']).round(1)
        
        
        weekday_patterns = df.groupby('day_of_week')['glucose_mgdl'].mean().round(1)
        
        return {
            'hourly_patterns': hourly_patterns.to_dict(),
            'weekday_patterns': weekday_patterns.to_dict(),
            'weekend_vs_weekday': {
                'weekend_mean': df[df['is_weekend']]['glucose_mgdl'].mean(),
                'weekday_mean': df[~df['is_weekend']]['glucose_mgdl'].mean()
            }
        }

# -----------------------------
# Visualization
# -----------------------------
class GlucoseVisualizer:
    """Creates visualizations for glucose analysis"""
    
    def __init__(self, config: GlucoseConfig):
        self.config = config
        plt.style.use('seaborn-v0_8')
        
    def create_comprehensive_report(self, df: pd.DataFrame, analysis: Dict, patient_id: str):
        """Creates comprehensive visual report"""
        
        fig = plt.figure(figsize=(20, 15))
        
        
        plt.subplot(3, 2, 1)
        self._plot_time_series(df, patient_id)
        
        
        plt.subplot(3, 2, 2)
        self._plot_period_distribution(df)
        
        
        plt.subplot(3, 2, 3)
        self._plot_heatmap(df)
        
        
        plt.subplot(3, 2, 4)
        self._plot_time_in_ranges(analysis['summary_stats']['time_in_ranges'])
        
        
        plt.subplot(3, 2, 5)
        self._plot_hourly_pattern(df)
        
        
        plt.subplot(3, 2, 6)
        self._plot_risk_summary(analysis['risk_assessment'])
        
        plt.tight_layout()
        plt.savefig(f'glucose_analysis_{patient_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_time_series(self, df: pd.DataFrame, patient_id: str):
        """Plots glucose time series"""
        
        plt.plot(df['timestamp'], df['glucose_mgdl'], alpha=0.7, linewidth=1)
                
        plt.axhline(y=self.config.MILD_HYPO, color='red', linestyle='--', alpha=0.7, label='Hypoglycemia')
        plt.axhline(y=self.config.NORMAL_MAX, color='green', linestyle='--', alpha=0.7, label='Normal Max')
        plt.axhline(y=self.config.MILD_HYPER, color='orange', linestyle='--', alpha=0.7, label='Hyperglycemia')
        plt.axhline(y=self.config.CRITICAL_HYPER, color='darkred', linestyle='--', alpha=0.7, label='Critical')
        
        plt.title(f'Time Series - Patient {patient_id}')
        plt.ylabel('Glucose (mg/dL)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_period_distribution(self, df: pd.DataFrame):
        """Plots distribution by time period"""
        
        period_data = [df[df['time_period'] == period]['glucose_mgdl'] for period in self.config.TIME_PERIODS.keys()]
        
        plt.boxplot(period_data, labels=self.config.TIME_PERIODS.keys())
        plt.title('Distribution by Time Period')
        plt.ylabel('Glucose (mg/dL)')
        plt.grid(True, alpha=0.3)
    
    def _plot_heatmap(self, df: pd.DataFrame):
        """Creates temporal heatmap"""
        
        df_heatmap = df.copy()
        df_heatmap['date'] = df_heatmap['timestamp'].dt.date
        df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
        
        pivot_table = df_heatmap.pivot_table(
            values='glucose_mgdl', 
            index='date', 
            columns='hour', 
            aggfunc='mean'
        ).fillna(0)
        
        plt.imshow(pivot_table, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='Glucose (mg/dL)')
        plt.title('Temporal Heatmap - Glucose by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Days')
    
    def _plot_time_in_ranges(self, time_in_ranges: Dict):
        """Plots time in different ranges"""
        
        ranges = list(time_in_ranges.keys())
        percentages = [time_in_ranges[r]['percentage'] for r in ranges]
        
        colors = ['red', 'orange', 'green', 'yellow', 'darkorange', 'darkred']
        plt.bar(ranges, percentages, color=colors)
        plt.title('Time in Each Glucose Range')
        plt.ylabel('Percentage of Time (%)')
        plt.xticks(rotation=45)
                
        for i, v in enumerate(percentages):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    def _plot_hourly_pattern(self, df: pd.DataFrame):
        """Plots average pattern by hour"""
        
        df['hour'] = df['timestamp'].dt.hour
        hourly_means = df.groupby('hour')['glucose_mgdl'].mean()
        
        plt.plot(hourly_means.index, hourly_means.values, marker='o')
        plt.axhline(y=self.config.MILD_HYPO, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=self.config.NORMAL_MAX, color='green', linestyle='--', alpha=0.5)
        plt.title('Average Pattern by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Average Glucose (mg/dL)')
        plt.grid(True, alpha=0.3)
    
    def _plot_risk_summary(self, risk_assessment: Dict):
        """Plots risk summary"""
        
        risks = risk_assessment['immediate_risks']
        if not risks:
            plt.text(0.5, 0.5, 'No immediate risks identified', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Risk Summary')
            return
        
        risk_text = "IDENTIFIED RISKS:\n\n"
        for risk in risks:
            risk_text += f"• {risk['risk']} ({risk['severity']})\n"
            risk_text += f"  {risk['description']}\n\n"
        
        plt.text(0.1, 0.9, risk_text, va='top', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Immediate Risk Summary')

# -----------------------------
# Main Function
# -----------------------------
def main():
    """Runs complete glucose pattern analysis"""
    
    logger.info("🏥 Starting Glucose Pattern Analysis")
    
    
    generator = GlucoseDataGenerator(CFG)
    patient_data = generator.generate_patient_data("P001", days=45)
    
    logger.info(f"📊 Data generated: {len(patient_data)} measurements")
    
    
    analyzer = GlucosePatternAnalyzer(CFG)
    analysis_results = analyzer.analyze_glucose_data(patient_data)
    
    
    visualizer = GlucoseVisualizer(CFG)
    visualizer.create_comprehensive_report(patient_data, analysis_results, "P001")
    
    
    output_file = f"glucose_analysis_P001_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(analysis_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    
    critical_alerts = analysis_results['risk_assessment']['immediate_risks']
    if critical_alerts:
        logger.warning("🚨 CRITICAL ALERTS IDENTIFIED:")
        for alert in critical_alerts:
            logger.warning(f"   • {alert['risk']}: {alert['description']}")
    
    logger.info(f"✅ Analysis completed! Results saved to: {output_file}")
    logger.info("📈 Visual report generated: glucose_analysis_P001.png")

if __name__ == "__main__":
    main()