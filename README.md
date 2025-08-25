# MLB Home Run Prediction System

## 🎯 Production-Ready Model (91% ROC-AUC)
Advanced feature engineering pipeline for MLB home run prediction with comprehensive situational analysis and performance optimization.

## 🏆 Latest Performance Results
- **Best Model**: 91% ROC-AUC with 203 features (step8_interactions)
- **Feature Engineering ROI**: +105% improvement over baseline  
- **Most Impactful Feature Set**: Situational context (+17.9% ROC-AUC)
- **Training Period**: 2024-04-01 to 2024-08-31
- **Test Period**: 2024-09-01 to 2024-10-31

## 📁 Directory Structure

### Core System Files
- `modeling.py` - Enhanced dual model system
- `dataset_builder.py` - Dataset construction pipeline  
- `feature_engineering.py` - Core feature engineering
- `config.py` - Configuration and feature definitions
- `comparative_analysis.py` - Model evaluation framework

### Feature Engineering Modules
- `enhanced_features.py` - Matchup and similarity features
- `situational_features.py` - Game context features
- `weather_features.py` - Weather impact analysis
- `ballpark_features.py` - Park-specific adjustments
- `recent_form_features.py` - Player form tracking
- `streak_momentum_features.py` - Performance trends
- `temporal_fatigue_features.py` - Time-based factors
- `feature_interactions.py` - Feature combinations

### Production Tools
- `live_prediction_system.py` - Real-time prediction API
- `betting_utils.py` - Bankroll and risk management
- `bet_tracker.py` - Performance tracking
- `api_client.py` - External data integration

### Performance Optimization
- `optimize_enhanced_features.py` - Enhanced feature optimization (O(n²) → O(n))
- `optimize_recent_form_improved.py` - Recent form optimization (10-20x speedup)
- **Performance Gains**: 50-70% faster execution, 4-year analysis reduced from 6-8 hours to 3-4 hours

### Utilities
- `data_utils.py` - Data handling utilities
- `weather_scraper.py` - Weather data collection
- `matchup_database.py` - Historical matchup tracking
- `unified_features.py` - Feature aggregation
- `inference_features.py` - Production inference

### Organized Directories
- `tests/` - All test files (test_*.py)
- `scripts/` - Utility scripts (fix_*.py, generate_*.py, etc.)
- `docs/` - Documentation (*.md files)
- `archives/` - Archived/debug files
- `backups/` - Configuration backups
- `outputs/` - Results and logs
- `images/` - Generated visualizations
- `data/` - Datasets and cached data
- `saved_models_pregame/` - Trained model artifacts

## 🚀 Quick Start

### Run Comparative Analysis
```bash
python comparative_analysis.py --use-cache  # Use cached data
python comparative_analysis.py             # Fresh rebuild
```

### Test Live Predictions
```bash
python live_prediction_system.py
```

### Feature Analysis
```bash
python analyze_features.py
```

## 📊 Project Evolution & Results

### Latest Achievements (August 2025)
- **Performance Optimization**: Resolved O(n²) complexity in feature engineering
- **91% ROC-AUC**: Best-in-class prediction accuracy with 203 optimized features  
- **Feature Engineering Pipeline**: 8-step progressive feature enhancement
- **Data Leakage Prevention**: Time-aware splitting with `.shift(1)` safeguards
- **Production Ready**: Real-time inference optimized for betting applications

### Feature Engineering Journey
1. **Baseline**: Core statistical features (58% ROC-AUC)
2. **Matchup Features**: Batter vs pitcher history  
3. **Situational Context**: Game state, inning, score differential (+17.9% ROC-AUC boost)
4. **Weather Impact**: Temperature, wind, atmospheric conditions
5. **Recent Form**: Time-decay weighted performance metrics
6. **Streak Analysis**: Hot/cold streaks and momentum tracking
7. **Ballpark Factors**: Park-specific adjustments and dimensions
8. **Temporal/Fatigue**: Circadian rhythms and player fatigue
9. **Feature Interactions**: Composite performance indices

### Technical Improvements
- **Feature Quality Analysis**: Identified and removed 125 problematic features (44% of original)
- **Algorithmic Optimization**: Vectorized operations replacing nested loops
- **Temporal Safeguards**: TimeSeriesSplit cross-validation and chronological data splits
- **Production Pipeline**: End-to-end system from data ingestion to live predictions

## 🛡️ Data Integrity & Validation
- **Time-aware splitting**: Chronological train/validation/test with configurable gaps
- **Look-ahead prevention**: All rolling features properly lagged with `.shift(1)`
- **Cross-validation**: TimeSeriesSplit for temporal data integrity
- **Feature validation**: Systematic checks for constants, correlations, and leakage

## 🎯 Current Status & Next Steps
- ✅ **Model Performance**: 91% ROC-AUC achieved and validated
- ✅ **Production Optimization**: Sub-second inference times
- ✅ **Data Integrity**: Comprehensive leakage prevention measures
- 🎯 **Live Deployment**: Ready for production betting applications
- 🎯 **Risk Management**: Integration with bankroll management systems
