#!/usr/bin/env python3
"""
Feature Analysis Script
Analyzes feature distributions to identify singular/low-variance features
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_features():
    # Load dataset
    df = pd.read_parquet('data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet')
    
    # Identify numeric columns (excluding identifiers)
    exclude_cols = ['date', 'batter', 'game_pk', 'home_team', 'away_team', 'ballpark', 'pitcher', 'hit_hr']
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in exclude_cols]
    
    print('FEATURE DISTRIBUTION ANALYSIS')
    print('=' * 80)
    print(f'Dataset shape: {df.shape}')
    print(f'Numeric features to analyze: {len(numeric_cols)}')
    print()
    
    feature_stats = []
    
    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna()
        
        if len(non_null) == 0:
            continue
        
        mean_val = non_null.mean()
        std_val = non_null.std()
        
        # Calculate coefficient of variation
        if mean_val != 0:
            coeff_var = std_val / abs(mean_val)
        else:
            coeff_var = float('inf') if std_val > 0 else 0
            
        stats = {
            'feature': col,
            'count': len(non_null),
            'unique_values': non_null.nunique(),
            'mean': mean_val,
            'std': std_val,
            'min': non_null.min(),
            'max': non_null.max(),
            'coeff_variation': coeff_var
        }
        
        feature_stats.append(stats)
    
    stats_df = pd.DataFrame(feature_stats)
    
    # CONSTANT FEATURES
    print('1. CONSTANT FEATURES (exactly 1 unique value):')
    print('-' * 80)
    constant_features = stats_df[stats_df['unique_values'] == 1].sort_values('feature')
    
    if len(constant_features) > 0:
        for _, row in constant_features.iterrows():
            print(f'  {row["feature"]:55} | value: {row["mean"]:10.4f}')
    else:
        print('  None found')
    
    print(f'\nTotal constant features: {len(constant_features)}')
    
    # NEARLY CONSTANT FEATURES
    print('\n2. NEARLY CONSTANT FEATURES (2-5 unique values):')
    print('-' * 80)
    nearly_constant = stats_df[
        (stats_df['unique_values'] > 1) & 
        (stats_df['unique_values'] <= 5)
    ].sort_values('unique_values')
    
    if len(nearly_constant) > 0:
        for _, row in nearly_constant.head(15).iterrows():  # Show first 15
            range_val = row['max'] - row['min']
            print(f'  {row["feature"]:55} | unique: {row["unique_values"]:2} | range: {range_val:8.4f}')
    else:
        print('  None found')
    
    print(f'\nTotal nearly constant features (2-5 unique): {len(nearly_constant)}')
    
    # LOW VARIANCE FEATURES
    print('\n3. LOW VARIANCE FEATURES (coefficient of variation < 0.1):')
    print('-' * 80)
    low_variance = stats_df[
        (stats_df['unique_values'] > 5) & 
        (stats_df['coeff_variation'] < 0.1) & 
        (stats_df['coeff_variation'] > 0)
    ].sort_values('coeff_variation')
    
    if len(low_variance) > 0:
        for _, row in low_variance.head(15).iterrows():  # Show first 15
            cv = row['coeff_variation']
            print(f'  {row["feature"]:55} | CV: {cv:8.6f} | std: {row["std"]:8.4f}')
    else:
        print('  None found')
    
    print(f'\nTotal low variance features: {len(low_variance)}')
    
    # SUMMARY
    total_problematic = len(constant_features) + len(nearly_constant) + len(low_variance)
    print('\n' + '='*80)
    print('SUMMARY:')
    print(f'  Constant features (1 unique):           {len(constant_features):3}')
    print(f'  Nearly constant features (2-5 unique):  {len(nearly_constant):3}')
    print(f'  Low variance features (CV < 0.1):       {len(low_variance):3}')
    print(f'  Total potentially problematic:          {total_problematic:3}')
    print(f'  Percentage of total features:           {total_problematic/len(stats_df)*100:5.1f}%')
    
    return stats_df, constant_features, nearly_constant, low_variance

if __name__ == "__main__":
    analyze_features()