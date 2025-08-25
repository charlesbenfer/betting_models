# MLB Home Run Prediction System - Setup Guide for Forks
========================================================

This guide walks you through setting up your own instance of the MLB home run prediction system from scratch.

## 🚀 Quick Start Overview

1. **Fork & Setup** → Clone repo, install dependencies
2. **Configure APIs** → Set up data and betting APIs  
3. **Train Models** → Build datasets and train your models
4. **Daily Workflow** → Use your trained models for predictions

---

## 📋 Prerequisites

### Required Accounts & API Keys
- **TheOdds API**: https://the-odds-api.com/ (for betting lines)
  - Free tier: 500 requests/month
  - Paid plans available for higher volume
- **Python 3.8+** with pip

### System Requirements
- **RAM**: 8GB+ recommended (for large dataset processing)
- **Storage**: 5GB+ free space (for cached data)
- **Internet**: Stable connection for API calls

---

## 🛠️ Step 1: Initial Setup

### Clone and Install
```bash
# Clone the repository
git clone <your-fork-url>
cd betting_models

# Install Python dependencies
pip install -r requirements.txt

# Install additional ML libraries
pip install pybaseball xgboost lightgbm scikit-learn
```

### Environment Configuration
```bash
# Create environment file
cp .env.template .env

# Edit .env file with your API keys
nano .env
```

Add to `.env`:
```bash
THEODDS_API_KEY=your_odds_api_key_here
```

---

## 🎯 Step 2: Train Your Own Models

### Option A: Recommended Method (Comparative Analysis - How Original Models Were Trained)
```bash
# Move the comparative analysis script back to root
mv archived_dev/comparative_analysis.py ./

# Run the same training process that achieved 91% ROC-AUC
python comparative_analysis.py

# This will:
# - Test 10 different feature combinations (baseline → full pipeline)
# - Train models with proper time-based splitting
# - Find the best performing model configuration
# - Save the best models automatically
# - Generate comprehensive performance reports

# Move back to archive when done
mv comparative_analysis.py archived_dev/
```

### Option B: Quick Single-Season Training
```bash
# Faster option with 2024 data only
mv archived_dev/comparative_analysis.py ./

# Edit the script to use smaller date range (optional)
python -c "
import re
with open('comparative_analysis.py', 'r') as f:
    content = f.read()

# Update to use 2024 data only for faster training
content = re.sub(
    r'start_date: str = \"2024-04-01\"',
    'start_date: str = \"2024-06-01\"',
    content
)
content = re.sub(
    r'end_date: str = \"2024-08-31\"',
    'end_date: str = \"2024-08-31\"',
    content
)

with open('comparative_analysis.py', 'w') as f:
    f.write(content)
"

# Run comparative analysis
python comparative_analysis.py

mv comparative_analysis.py archived_dev/
```

### Option C: Manual Training (Advanced Users)
```bash
# Direct model training (skip comparative analysis)
python -c "
from modeling import EnhancedDualModelSystem
from dataset_builder import PregameDatasetBuilder

# Build dataset with same parameters as comparative analysis
builder = PregameDatasetBuilder(start_date='2024-04-01', end_date='2024-08-31')
dataset = builder.build_dataset(force_rebuild=True)

# Train models
model_system = EnhancedDualModelSystem()
results = model_system.fit(
    dataset, 
    splitting_strategy='time_based',
    test_start_date='2024-09-01',
    cross_validate=True
)

# Save models
model_system.save()
print('Training complete!')
"
```

---

## ⚙️ Step 3: Verify Your Setup

### Test Model Training
```bash
# Quick test with small dataset
python -c "
from dataset_builder import PregameDatasetBuilder
from modeling import EnhancedDualModelSystem

# Test with 1 month of data
builder = PregameDatasetBuilder(start_date='2024-08-01', end_date='2024-08-31')
test_data = builder.build_dataset()

if len(test_data) > 100:
    print(f'✅ Dataset building works: {len(test_data)} samples')
    
    # Quick model test
    model = EnhancedDualModelSystem()
    results = model.fit(test_data, cross_validate=False)
    
    if results['core_val_metrics']:
        print('✅ Model training works')
    else:
        print('❌ Model training failed')
else:
    print('❌ Dataset building failed')
"
```

### Test API Connectivity
```bash
# Test odds API
python -c "
from api_client import SafeAPIClient
from config import config

client = SafeAPIClient(config.THEODDS_API_KEY)
if client.test_connection():
    print('✅ Odds API connected')
else:
    print('❌ Odds API failed - check your API key')
"
```

### Test Live Predictions
```bash
# Test prediction system
python -c "
from live_prediction_system import create_live_system

try:
    system = create_live_system()
    print('✅ Live prediction system ready')
except Exception as e:
    print(f'❌ Setup issue: {e}')
"
```

---

## 🔄 Step 4: Your Daily Workflow

Once setup is complete, your daily routine is simple:

### Morning Data Update (7:00 AM)
```bash
# Pull fresh MLB data (last 45 days)
python fetch_recent_data.py
```

### Generate Predictions (8:00 AM, 12:00 PM, 4:00 PM)
```bash
# Basic predictions
python live_prediction_system.py

# Conservative betting opportunities
python live_prediction_system.py --min-ev 0.05 --min-confidence 0.70

# More aggressive opportunities
python live_prediction_system.py --min-ev 0.02 --min-confidence 0.60
```

---

## 🎛️ Configuration Options

### Model Training Parameters
Edit these in your training scripts:

```python
# In your training code
model_system = EnhancedDualModelSystem()
results = model_system.fit(
    dataset,
    splitting_strategy='time_based',  # 'time_based', 'random', or 'seasonal'
    test_size=0.2,                   # 20% for testing
    val_size=0.1,                    # 10% for validation
    gap_days=7,                      # 7-day gap to prevent leakage
    cross_validate=True,             # Enable cross-validation
    cv_folds=5                       # 5-fold CV
)
```

### Prediction Thresholds
Adjust in `config.py`:

```python
# Betting thresholds
MIN_EV_THRESHOLD = 0.03        # 3% minimum expected value
MIN_PROB_THRESHOLD = 0.10      # 10% minimum HR probability
MIN_CONFIDENCE_THRESHOLD = 0.65 # 65% minimum confidence
```

### Date Ranges for Training
```python
# Recent season only (faster training)
PregameDatasetBuilder(start_date='2024-04-01', end_date='2024-09-30')

# Multi-season (better performance)
PregameDatasetBuilder(start_date='2021-01-01', end_date='2024-09-30')

# Custom range
PregameDatasetBuilder(start_date='2023-04-01', end_date='2024-08-31')
```

---

## 📊 Expected Performance

### Training Time
- **Single season**: 30-60 minutes
- **Multi-season**: 2-4 hours
- **Full historical**: 4-8 hours

### Model Performance Targets
- **Baseline ROC-AUC**: 0.60-0.70 (basic features)
- **Enhanced ROC-AUC**: 0.80-0.90 (full pipeline)
- **Our best result**: 0.91 ROC-AUC

### Daily Runtime
- **Data fetching**: 5-10 minutes
- **Predictions**: 30-60 seconds

---

## 🔧 Troubleshooting

### Common Issues

**"No data returned from pybaseball"**
```bash
# Clear pybaseball cache and retry
python -c "import pybaseball as pb; pb.cache.purge()"
python fetch_recent_data.py
```

**"Model files not found"**
- Ensure you completed model training step
- Check that `saved_models_pregame/` directory has .joblib files

**"API rate limit exceeded"**
- Check your TheOdds API usage
- Wait for rate limit reset
- Consider upgrading API plan

**"Features missing during prediction"**
- Ensure you ran `fetch_recent_data.py` first
- Check that data files exist in `data/processed/`

### Getting Help

1. **Check logs**: Look in generated log files for error details
2. **Test components**: Run individual test scripts from `archived_dev/tests/`
3. **Restore development tools**: Copy files back from `archived_dev/` as needed

---

## 🚀 Advanced Customization

### Add New Features
1. Create new feature module (e.g., `my_custom_features.py`)
2. Add import to `dataset_builder.py`
3. Add feature calculation step in pipeline
4. Retrain models

### Modify Model Architecture
1. Edit `modeling.py` → `BaseModelComponent` class
2. Add new algorithms or change hyperparameters
3. Retrain and compare performance

### Integrate New Data Sources
1. Create new API client (follow `api_client.py` pattern)
2. Add data processing in `dataset_builder.py`
3. Update feature engineering pipeline

---

## ⚠️ Important Notes

### Data Usage
- **pybaseball**: Free but has rate limits
- **TheOdds API**: Paid service, monitor usage
- **Large datasets**: Can take significant time/storage

### Legal Considerations
- Ensure sports betting is legal in your jurisdiction
- This system is for educational/research purposes
- Always gamble responsibly if using for betting

### Performance
- Results may vary based on data quality and time periods
- Model performance can degrade over time (concept drift)
- Regular retraining recommended

---

## 📈 Success Metrics

Your setup is working correctly when:
- ✅ Models achieve >0.80 ROC-AUC on test data
- ✅ Daily predictions complete in <2 minutes
- ✅ System finds 2-5 betting opportunities per day
- ✅ Predictions are calibrated (probability ≈ actual rate)

---

This system represents months of development and optimization. With proper setup, you should achieve similar performance to our 91% ROC-AUC results. Good luck with your implementation!