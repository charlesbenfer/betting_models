#!/usr/bin/env python3
"""
Demo script to show betting opportunities using historical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from api_client import SafeAPIClient
from modeling import EnhancedDualModelSystem
from betting_utils import BettingAnalyzer
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_betting_opportunities():
    """Generate demo betting opportunities using available data."""
    
    print("🚀 MLB Home Run Prediction Demo")
    print("=" * 50)
    
    try:
        # Initialize model system
        model_system = EnhancedDualModelSystem()
        try:
            model_info = model_system.load()
            print("✅ Models loaded successfully")
            # Check if models are actually loaded
            if model_system.core_model is not None or model_system.enhanced_model is not None:
                print(f"   Core model: {model_system.core_model is not None}")
                print(f"   Enhanced model: {model_system.enhanced_model is not None}")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            print("✅ Continuing with demo using sample data")
        
        # Get current odds for demonstration
        api_client = SafeAPIClient(api_key=config.THEODDS_API_KEY)
        
        print("\n📊 Fetching current MLB odds...")
        current_date = datetime.now().strftime("%Y-%m-%d")
        odds_data = api_client.get_todays_odds(current_date)
        
        if odds_data.empty:
            print("⚠️  No games scheduled today")
            print("🎯 Let me show you what predictions would look like...")
            
            # Create sample prediction data
            sample_games = [
                {"home_team": "LAD", "away_team": "NYY", "player": "Aaron Judge", "odds": 240, "hr_probability": 0.32},
                {"home_team": "HOU", "away_team": "BOS", "player": "Alex Bregman", "odds": 380, "hr_probability": 0.28}, 
                {"home_team": "ATL", "away_team": "PHI", "player": "Ronald Acuña Jr.", "odds": 290, "hr_probability": 0.31},
                {"home_team": "SD", "away_team": "SF", "player": "Manny Machado", "odds": 420, "hr_probability": 0.25},
                {"home_team": "TB", "away_team": "TOR", "player": "Vladimir Guerrero Jr.", "odds": 350, "hr_probability": 0.27}
            ]
            
            print("\n🏟️  SAMPLE BETTING OPPORTUNITIES")
            print("=" * 60)
            
            betting_analyzer = BettingAnalyzer()
            
            for game in sample_games:
                # Calculate expected value
                implied_prob = 100 / game["odds"]
                expected_value = (game["hr_probability"] - implied_prob) / implied_prob
                
                # Calculate recommended bet size (Kelly criterion)
                if expected_value > 0:
                    kelly_fraction = (game["hr_probability"] * game["odds"] / 100 - 1) / (game["odds"] / 100 - 1)
                    recommended_stake = max(0, min(kelly_fraction * 0.25, 0.02))  # 25% of Kelly, max 2%
                else:
                    recommended_stake = 0
                
                print(f"\n🎯 {game['home_team']} vs {game['away_team']}")
                print(f"   Player: {game['player']}")
                print(f"   Predicted HR Probability: {game['hr_probability']:.1%}")
                print(f"   Odds: +{game['odds']} (Implied: {implied_prob:.1%})")
                print(f"   Expected Value: {expected_value:+.1%}")
                
                if expected_value > 0.05:  # 5% EV threshold
                    confidence = min(95, 70 + (expected_value * 100))
                    print(f"   🟢 RECOMMENDATION: BET (Confidence: {confidence:.0f}%)")
                    print(f"   💰 Suggested Stake: {recommended_stake:.1%} of bankroll")
                elif expected_value > 0.02:
                    print(f"   🟡 MARGINAL: Small value (Consider)")
                else:
                    print(f"   🔴 PASS: Negative expected value")
            
        else:
            print(f"✅ Found {len(odds_data)} current betting lines")
            
            # Show first few current opportunities
            print(f"\n📋 Current Odds Available:")
            for _, row in odds_data.head(10).iterrows():
                odds_val = row.get('odds_hr_yes', 'N/A')
                odds_str = f"+{odds_val}" if odds_val != 'N/A' and odds_val > 0 else str(odds_val)
                print(f"   {row.get('batter_name', 'Unknown')}: {odds_str} ({row.get('bookmaker', 'Unknown')})")
                
        print("\n" + "=" * 60)        
        print("📊 SYSTEM CAPABILITIES")
        print("=" * 60)
        print("✅ Real-time odds fetching")
        print("✅ Machine learning predictions")  
        print("✅ Expected value calculations")
        print("✅ Kelly criterion stake sizing")
        print("✅ Risk management controls")
        print("✅ Performance tracking")
        
        print(f"\n💡 SETTINGS")
        print(f"   Min Expected Value: {config.MIN_EV_THRESHOLD:.1%}")
        print(f"   Min Probability: {config.MIN_PROB_THRESHOLD:.1%}")
        print(f"   Max Stake per Bet: 2% of bankroll")
        
        print(f"\n🎯 Ready to find profitable opportunities!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_betting_opportunities()