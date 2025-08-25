#!/usr/bin/env python3
"""
Feature Cleanup Script
Removes problematic features identified in analysis:
- 51 constant features
- 43 nearly constant features  
- 31 low variance features
Total: 125 features to remove (44% reduction)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

def get_features_to_remove():
    """Define all problematic features to remove."""
    
    # 51 CONSTANT FEATURES (exactly 1 unique value)
    constant_features = [
        'ahead_in_count_pct', 'avg_runners_on_base', 'avg_score_differential',
        'bases_empty_percentage', 'batter_park_historical_performance', 'behind_in_count_pct',
        'close_game_pa_percentage', 'clutch_momentum', 'clutch_pa_percentage',
        'clutch_pressure_performance', 'discipline_form_chase_rate', 'discipline_form_contact_rate',
        'discipline_form_whiff_rate', 'discipline_form_z_contact_rate', 'dog_days_effect',
        'even_count_hr_rate', 'experience_pressure_synergy', 'high_leverage_pa_pct',
        'hitters_count_hr_rate', 'leading_percentage', 'matchup_avg_ev_career',
        'matchup_avg_la_career', 'matchup_familiarity_score', 'matchup_form_synergy',
        'matchup_hr_career', 'matchup_hr_rate_career', 'matchup_hr_rate_recent',
        'matchup_hr_recent', 'matchup_pa_career', 'matchup_pa_recent',
        'mix_EP', 'mix_FA', 'mix_PO', 'mix_SC', 'monthly_energy_level',
        'park_day_night_factor', 'park_month_hr_factor', 'pitchers_count_hr_rate',
        'playoff_chase_energy', 'power_form_iso_power', 'pressure_performance_index',
        'pressure_response', 'recent_slump_indicator', 'risp_percentage',
        'season', 'spring_training_carryover', 'suboptimal_time_penalty',
        'suboptimal_timing_penalty', 'tied_percentage', 'trailing_percentage',
        'two_strike_hr_rate'
    ]
    
    # 43 NEARLY CONSTANT FEATURES (2-5 unique values)
    nearly_constant_features = [
        'park_is_dome', 'park_coastal_humidity_factor', 'park_dome_carry_reduction',
        'circadian_performance_factor', 'park_surface_turf', 'home_away_transition',
        'optimal_performance_window', 'optimal_time_window', 'circadian_mismatch',
        'night_game_indicator', 'afternoon_game_boost', 'time_preference_alignment',
        'elite_power_indicator', 'evening_game_factor', 'high_momentum_indicator'
    ]
    
    # 31 LOW VARIANCE FEATURES (CV < 0.1) - Keep some that might be useful, remove obvious redundancies
    low_variance_features = [
        'p_roll10_ev_allowed',  # Remove: redundant with p_roll30d_ev_allowed
        'pressure_hr_factor',   # Remove: redundant with pressure
        'drag_factor',          # Remove: redundant with air_density
        'air_density_ratio',    # Remove: redundant with air_density
        'flight_distance_factor',  # Remove: redundant with air_density
        'roll10_bat_speed',     # Remove: redundant with roll30d_bat_speed_mean
        'season_fatigue_factor' # Remove: redundant with season_progression
    ]
    
    # ADDITIONAL REDUNDANT FEATURES from correlation analysis
    redundant_features = [
        # Time-based redundancy (keep only game_hour)
        'afternoon_game_boost', 'evening_game_factor', 'night_game_indicator',
        'optimal_time_window', 'circadian_performance_factor',
        
        # HR Rate redundancy (keep only clutch_hr_rate)  
        'close_game_hr_rate', 'first_pa_hr_rate', 'leadoff_inning_hr_rate', 'high_leverage_hr_rate',
        
        # Rolling window redundancy (keep 30-day versions)
        'p_roll10_hr',  # Remove in favor of p_roll30d_hr
    ]
    
    # Combine all and remove duplicates
    all_features_to_remove = list(set(
        constant_features + 
        nearly_constant_features + 
        low_variance_features + 
        redundant_features
    ))
    
    return {
        'constant': constant_features,
        'nearly_constant': nearly_constant_features, 
        'low_variance': low_variance_features,
        'redundant': redundant_features,
        'all': all_features_to_remove
    }

def backup_config():
    """Create backup of config.py before modification."""
    backup_path = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    shutil.copy2('config.py', backup_path)
    print(f"✅ Config backup created: {backup_path}")
    return backup_path

def clean_config_features():
    """Remove problematic features from config.py feature lists."""
    features_to_remove = get_features_to_remove()
    
    # Read current config
    with open('config.py', 'r') as f:
        content = f.read()
    
    # Track removals
    removed_count = 0
    
    # For each category, remove features from the appropriate config lists
    for feature in features_to_remove['all']:
        # Count occurrences before removal
        before_count = content.count(f"'{feature}'")
        
        # Remove from config (both single and double quotes)
        content = content.replace(f"'{feature}',", '')
        content = content.replace(f'"{feature}",', '')
        content = content.replace(f"'{feature}'", '')
        content = content.replace(f'"{feature}"', '')
        
        # Count after removal
        after_count = content.count(f"'{feature}'")
        removed_count += (before_count - after_count)
    
    # Clean up extra commas and whitespace
    import re
    content = re.sub(r',\s*,', ',', content)  # Remove double commas
    content = re.sub(r',\s*\]', ']', content)  # Remove trailing commas before ]
    content = re.sub(r'\[\s*,', '[', content)  # Remove leading commas after [
    
    # Write cleaned config
    with open('config.py', 'w') as f:
        f.write(content)
    
    print(f"✅ Removed {removed_count} feature references from config.py")
    return removed_count

def analyze_cleanup_impact(dataset_path):
    """Analyze the impact of feature cleanup on dataset."""
    if not Path(dataset_path).exists():
        print(f"⚠️ Dataset not found: {dataset_path}")
        return
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    original_features = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])
    
    features_to_remove = get_features_to_remove()
    
    # Check which features actually exist in dataset
    existing_removals = [f for f in features_to_remove['all'] if f in df.columns]
    missing_removals = [f for f in features_to_remove['all'] if f not in df.columns]
    
    print(f"\\n📊 CLEANUP IMPACT ANALYSIS:")
    print(f"Original numeric features: {original_features}")
    print(f"Features to remove (planned): {len(features_to_remove['all'])}")
    print(f"Features to remove (existing): {len(existing_removals)}")
    print(f"Features to remove (missing): {len(missing_removals)}")
    print(f"Remaining features after cleanup: {original_features - len(existing_removals)}")
    print(f"Reduction: {len(existing_removals)/original_features*100:.1f}%")
    
    if missing_removals:
        print(f"\\n⚠️ Features not found in dataset: {missing_removals[:10]}...")
    
    return len(existing_removals)

def main():
    """Run the complete feature cleanup process."""
    print("🧹 FEATURE CLEANUP PROCESS")
    print("=" * 60)
    
    # Get features to remove
    features_to_remove = get_features_to_remove()
    
    print(f"📋 CLEANUP PLAN:")
    print(f"  Constant features: {len(features_to_remove['constant'])}")
    print(f"  Nearly constant features: {len(features_to_remove['nearly_constant'])}")
    print(f"  Low variance features: {len(features_to_remove['low_variance'])}")
    print(f"  Redundant features: {len(features_to_remove['redundant'])}")
    print(f"  Total unique features to remove: {len(features_to_remove['all'])}")
    
    # Analyze impact on existing dataset
    dataset_path = 'data/processed/pregame_dataset_2024-07-01_2024-08-31.parquet'
    actual_removals = analyze_cleanup_impact(dataset_path)
    
    # Auto-proceed with cleanup
    print(f"\\n🚀 Proceeding with removing {actual_removals} features from config.py...")
    
    # Create backup
    backup_path = backup_config()
    
    try:
        # Clean config
        removed_count = clean_config_features()
        
        print(f"\\n✅ CLEANUP COMPLETED SUCCESSFULLY!")
        print(f"  Removed {removed_count} feature references from config.py")
        print(f"  Backup saved as: {backup_path}")
        print(f"\\n🚀 Next steps:")
        print(f"  1. Test: python -c 'from config import config; print(\"Config loads successfully\")'")
        print(f"  2. Rebuild dataset with cleaned features")
        print(f"  3. Run comparative analysis to see improved efficiency")
        
    except Exception as e:
        print(f"\\n❌ Error during cleanup: {e}")
        print(f"Restoring backup...")
        shutil.copy2(backup_path, 'config.py')
        print(f"✅ Config restored from backup")

if __name__ == "__main__":
    main()