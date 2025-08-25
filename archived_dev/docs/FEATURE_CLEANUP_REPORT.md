# Feature Quality Analysis & Cleanup Recommendations

## Executive Summary

Our analysis of 283 numeric features revealed **severe redundancy and quality issues**:

- **51 features (18%) are completely constant** - providing zero information
- **43 features (15%) are nearly constant** - minimal variation
- **31 features (11%) have very low variance** - coefficient of variation < 0.1  
- **111 feature pairs have perfect correlation (>0.99)** - complete redundancy
- **231 feature pairs have high correlation (>0.9)** - substantial redundancy

**Total problematic features: ~125 (44% of all features)**

## Problems Identified

### 1. Constant Features (51 features - 0 information value)

**Matchup Features (9/17 constant):**
- All career matchup stats: `matchup_pa_career`, `matchup_hr_career`, `matchup_hr_rate_career`, etc.
- Recent matchup stats: `matchup_pa_recent`, `matchup_hr_recent`, `matchup_hr_rate_recent`
- All returning 0.0 - indicates insufficient historical matchup data

**Situational Features (17/33 constant):**
- Count-based features: `ahead_in_count_pct`, `behind_in_count_pct`, `even_count_hr_rate`, etc.
- Score situation: `avg_score_differential`, `leading_percentage`, `tied_percentage`, `trailing_percentage`
- Base runners: `avg_runners_on_base`, `risp_percentage`, `bases_empty_percentage` 
- All returning fixed values - indicates missing situational data

**Recent Form Features (6/24 constant):**
- All discipline features: `discipline_form_chase_rate`, `discipline_form_contact_rate`, etc. 
- All returning 0.5 - indicates approximation/placeholder values from our optimizer

### 2. Perfect Correlations (111 pairs - complete redundancy)

**Time-based redundancy:**
- `game_hour` perfectly correlates with 8+ time-related features
- `season_fatigue_factor` = `season_progression` (1.0 correlation)
- All circadian/time features are variants of the same information

**Weather redundancy:**
- `air_density` = `air_density_ratio` = `drag_factor` (all 1.0 correlation)
- `pressure` = `pressure_hr_factor` (1.0 correlation)

**Rolling window redundancy:**
- `roll10_*` features nearly perfectly correlate with `roll30d_*` versions
- `p_roll10_hr` = `p_roll30d_hr` = `p_roll30d_hr_rate` (1.0 correlation)

**Leverage/situation redundancy:**
- Multiple HR rate features are identical: `clutch_hr_rate` = `close_game_hr_rate` = `first_pa_hr_rate` = `leadoff_inning_hr_rate` = `high_leverage_hr_rate`

## Impact on Model Performance

### Current Feature Counts (with redundancy):
```
baseline_core:         37 features
step4_recent_form:    131 features  
step8_interactions:   275 features
```

### Estimated Clean Feature Counts:
```
baseline_core:         ~25 features (-12 redundant)
step4_recent_form:     ~75 features (-56 redundant) 
step8_interactions:   ~150 features (-125 redundant)
```

## Recommendations

### Priority 1: Remove Constant Features (51 features)
These provide zero information and should be removed immediately:

**From MATCHUP_FEATURES (remove 9):**
```python
REMOVE = [
    'matchup_pa_career', 'matchup_hr_career', 'matchup_hr_rate_career',
    'matchup_avg_ev_career', 'matchup_avg_la_career', 'matchup_pa_recent', 
    'matchup_hr_recent', 'matchup_hr_rate_recent', 'matchup_familiarity_score'
]
```

**From SITUATIONAL_FEATURES (remove 17):**
```python  
REMOVE = [
    'ahead_in_count_pct', 'behind_in_count_pct', 'even_count_hr_rate',
    'hitters_count_hr_rate', 'pitchers_count_hr_rate', 'two_strike_hr_rate',
    'avg_runners_on_base', 'risp_percentage', 'bases_empty_percentage',
    'avg_score_differential', 'leading_percentage', 'tied_percentage', 'trailing_percentage',
    'close_game_pa_percentage', 'clutch_pa_percentage', 'high_leverage_pa_pct',
    'pressure_performance_index'
]
```

**From RECENT_FORM_FEATURES (remove 6):**
```python
REMOVE = [
    'discipline_form_chase_rate', 'discipline_form_contact_rate', 
    'discipline_form_whiff_rate', 'discipline_form_z_contact_rate',
    'power_form_iso_power', 'recent_slump_indicator'
]
```

### Priority 2: Remove Redundant Features (choose 1 from each correlated group)

**Time/Circadian group (keep only `game_hour`):**
- Remove: `circadian_performance_factor`, `afternoon_game_boost`, `evening_game_factor`, `optimal_time_window`, `night_game_indicator`, etc.

**Weather/Physics group (keep only `air_density`):**
- Remove: `air_density_ratio`, `drag_factor`, `flight_distance_factor`

**Rolling Windows group (keep 30-day versions):**
- Remove most `roll10_*` features, keep `roll30d_*` variants
- Remove: `p_roll10_hr`, `p_roll10_ev_allowed` (keep 30-day versions)

**HR Rate group (keep only `clutch_hr_rate`):**
- Remove: `close_game_hr_rate`, `first_pa_hr_rate`, `leadoff_inning_hr_rate`, `high_leverage_hr_rate`

### Priority 3: Feature Engineering Fixes

**Fix data issues causing constants:**
1. Implement proper count situation tracking for situational features
2. Add real matchup history data or remove matchup features entirely  
3. Fix discipline form calculations to provide actual variation

**Improve feature orthogonality:**
1. Ensure rolling windows use different time periods (e.g., 10-game vs 30-day vs 60-day)
2. Create composite indices instead of multiple similar metrics
3. Add feature interaction terms that capture unique relationships

## Expected Benefits

**Model training efficiency:**
- ~45% fewer features = faster training and prediction
- Reduced overfitting risk
- Better generalization performance

**Interpretability:**
- Cleaner feature importance rankings
- More meaningful coefficient analysis  
- Easier model debugging

**Maintenance:**
- Simpler pipeline with fewer failure points
- Reduced computational costs
- More focused feature engineering efforts

## Implementation Plan

1. **Phase 1:** Remove all constant features from config.py lists
2. **Phase 2:** Remove perfectly correlated redundant features  
3. **Phase 3:** Test comparative analysis with cleaned feature sets
4. **Phase 4:** Fix underlying data issues causing constants
5. **Phase 5:** Re-engineer composite features to replace removed ones

This cleanup will result in a more robust, efficient, and interpretable model while maintaining (or improving) predictive performance.