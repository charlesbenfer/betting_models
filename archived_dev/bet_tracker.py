"""
Bet Tracking System
==================

Track betting performance over time with detailed analytics.
"""

import pandas as pd
import numpy as np
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import requests
from live_prediction_system import LivePredictionSystem

logger = logging.getLogger(__name__)

@dataclass
class BetRecord:
    """Record of a single bet placed."""
    bet_id: str
    date: str
    player_name: str
    model_probability: float
    odds: float
    bet_amount: float
    expected_value: float
    kelly_fraction: float
    confidence_score: float
    bookmaker: str
    bet_type: str = "home_run_yes"
    status: str = "pending"  # pending, won, lost, void
    actual_result: Optional[bool] = None
    payout: Optional[float] = None
    profit_loss: Optional[float] = None
    game_id: Optional[str] = None
    notes: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.bet_id:
            self.bet_id = f"{self.date}_{self.player_name.replace(' ', '_')}_{self.timestamp[-8:]}"

class BetTracker:
    """Main bet tracking system."""
    
    def __init__(self, db_path: str = "betting_tracker.db", 
                 max_daily_risk: float = 100.0,
                 bankroll: float = 1000.0):
        self.db_path = Path(db_path)
        self.max_daily_risk = max_daily_risk
        self.bankroll = bankroll
        self.conn = None
        self._setup_database()
        
    def _setup_database(self):
        """Initialize SQLite database for bet tracking."""
        self.conn = sqlite3.connect(self.db_path)
        
        # Create bets table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                bet_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                player_name TEXT NOT NULL,
                model_probability REAL NOT NULL,
                odds REAL NOT NULL,
                bet_amount REAL NOT NULL,
                expected_value REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                confidence_score REAL NOT NULL,
                bookmaker TEXT NOT NULL,
                bet_type TEXT DEFAULT 'home_run_yes',
                status TEXT DEFAULT 'pending',
                actual_result INTEGER,  -- 0/1 for False/True, NULL for pending
                payout REAL,
                profit_loss REAL,
                game_id TEXT,
                notes TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Create daily summary table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_bets INTEGER DEFAULT 0,
                total_wagered REAL DEFAULT 0,
                total_profit_loss REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                actual_ev REAL DEFAULT 0,
                bankroll_change REAL DEFAULT 0,
                notes TEXT
            )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def place_bet(self, bet_record: BetRecord) -> bool:
        """Record a new bet."""
        try:
            # Check daily risk limit
            daily_risk = self.get_daily_risk(bet_record.date)
            if daily_risk + bet_record.bet_amount > self.max_daily_risk:
                logger.warning(f"Bet exceeds daily risk limit: ${daily_risk + bet_record.bet_amount:.2f} > ${self.max_daily_risk:.2f}")
                return False
            
            # Insert bet record
            bet_dict = asdict(bet_record)
            columns = list(bet_dict.keys())
            placeholders = ["?" * len(columns)]
            
            # Handle boolean conversion for actual_result
            if bet_dict['actual_result'] is not None:
                bet_dict['actual_result'] = int(bet_dict['actual_result'])
            
            self.conn.execute(
                f"INSERT INTO bets ({','.join(columns)}) VALUES ({','.join(['?'] * len(columns))})",
                list(bet_dict.values())
            )
            self.conn.commit()
            
            logger.info(f"Bet placed: {bet_record.player_name} ${bet_record.bet_amount:.2f} @ {bet_record.odds:+.0f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place bet: {e}")
            return False
    
    def update_bet_result(self, bet_id: str, won: bool, actual_payout: float = None) -> bool:
        """Update bet with actual result."""
        try:
            # Get bet details
            bet = self.conn.execute(
                "SELECT bet_amount, odds FROM bets WHERE bet_id = ?", (bet_id,)
            ).fetchone()
            
            if not bet:
                logger.error(f"Bet not found: {bet_id}")
                return False
            
            bet_amount, odds = bet
            
            # Calculate payout and profit/loss
            if won:
                if actual_payout is not None:
                    payout = actual_payout
                else:
                    # Calculate expected payout from odds
                    if odds >= 0:
                        payout = bet_amount * (odds / 100.0)
                    else:
                        payout = bet_amount * (100.0 / abs(odds))
                
                profit_loss = payout
                status = "won"
            else:
                payout = 0.0
                profit_loss = -bet_amount
                status = "lost"
            
            # Update database
            self.conn.execute("""
                UPDATE bets 
                SET status = ?, actual_result = ?, payout = ?, profit_loss = ?
                WHERE bet_id = ?
            """, (status, int(won), payout, profit_loss, bet_id))
            
            self.conn.commit()
            
            logger.info(f"Bet updated: {bet_id} -> {status} (P&L: ${profit_loss:+.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update bet result: {e}")
            return False
    
    def get_daily_risk(self, date: str) -> float:
        """Get total amount wagered on a specific date."""
        result = self.conn.execute(
            "SELECT COALESCE(SUM(bet_amount), 0) FROM bets WHERE date = ?", (date,)
        ).fetchone()
        return result[0] if result else 0.0
    
    def get_performance_summary(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get comprehensive performance summary."""
        # Build date filter
        date_filter = ""
        params = []
        
        if start_date:
            date_filter += " AND date >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND date <= ?"
            params.append(end_date)
        
        # Get overall statistics
        query = f"""
            SELECT 
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status != 'pending' THEN 1 END) as settled_bets,
                COUNT(CASE WHEN status = 'won' THEN 1 END) as wins,
                COUNT(CASE WHEN status = 'lost' THEN 1 END) as losses,
                COALESCE(SUM(bet_amount), 0) as total_wagered,
                COALESCE(SUM(profit_loss), 0) as total_profit_loss,
                COALESCE(AVG(model_probability), 0) as avg_model_prob,
                COALESCE(AVG(expected_value), 0) as avg_expected_value,
                COALESCE(AVG(confidence_score), 0) as avg_confidence
            FROM bets 
            WHERE 1=1 {date_filter}
        """
        
        stats = self.conn.execute(query, params).fetchone()
        
        if not stats or stats[0] == 0:
            return {"error": "No bets found in specified period"}
        
        total_bets, settled_bets, wins, losses, total_wagered, total_pl, avg_prob, avg_ev, avg_conf = stats
        
        # Calculate metrics
        win_rate = wins / settled_bets if settled_bets > 0 else 0
        roi = total_pl / total_wagered if total_wagered > 0 else 0
        actual_ev = total_pl / settled_bets if settled_bets > 0 else 0
        
        # Get daily breakdown
        daily_query = f"""
            SELECT 
                date,
                COUNT(*) as bets,
                SUM(bet_amount) as wagered,
                COALESCE(SUM(profit_loss), 0) as profit_loss,
                COUNT(CASE WHEN status = 'won' THEN 1 END) as wins,
                COUNT(CASE WHEN status != 'pending' THEN 1 END) as settled
            FROM bets 
            WHERE 1=1 {date_filter}
            GROUP BY date
            ORDER BY date
        """
        
        daily_data = self.conn.execute(daily_query, params).fetchall()
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": len(daily_data)
            },
            "overall": {
                "total_bets": total_bets,
                "settled_bets": settled_bets,
                "pending_bets": total_bets - settled_bets,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_wagered": total_wagered,
                "total_profit_loss": total_pl,
                "roi": roi,
                "actual_ev_per_bet": actual_ev,
                "avg_model_probability": avg_prob,
                "avg_expected_value": avg_ev,
                "avg_confidence_score": avg_conf
            },
            "daily_data": [
                {
                    "date": row[0],
                    "bets": row[1],
                    "wagered": row[2],
                    "profit_loss": row[3],
                    "wins": row[4],
                    "settled": row[5],
                    "win_rate": row[4] / row[5] if row[5] > 0 else 0
                }
                for row in daily_data
            ]
        }
    
    def generate_performance_report(self, days: int = 7) -> str:
        """Generate a formatted performance report."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        summary = self.get_performance_summary(start_date, end_date)
        
        if "error" in summary:
            return f"No betting data found for the last {days} days."
        
        overall = summary["overall"]
        
        report = f"""
{'='*60}
BETTING PERFORMANCE REPORT ({days} days)
{'='*60}

OVERALL STATISTICS:
  Total Bets Placed: {overall['total_bets']:,}
  Settled Bets: {overall['settled_bets']:,}
  Pending Bets: {overall['pending_bets']:,}
  
PERFORMANCE:
  Win Rate: {overall['win_rate']:.1%} ({overall['wins']}/{overall['settled_bets']})
  Total Wagered: ${overall['total_wagered']:,.2f}
  Total P&L: ${overall['total_profit_loss']:+,.2f}
  ROI: {overall['roi']:+.1%}
  
MODEL ACCURACY:
  Avg Model Probability: {overall['avg_model_probability']:.1%}
  Avg Expected Value: {overall['avg_expected_value']:+.3f}
  Actual EV per Bet: ${overall['actual_ev_per_bet']:+.2f}
  Avg Confidence Score: {overall['avg_confidence_score']:.1f}/10

DAILY BREAKDOWN:
"""
        
        for day in summary["daily_data"]:
            report += f"  {day['date']}: {day['bets']} bets, ${day['wagered']:.0f} wagered, ${day['profit_loss']:+.0f} P&L, {day['win_rate']:.0%} WR\n"
        
        return report
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export all bet data to CSV."""
        if filename is None:
            filename = f"betting_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        query = """
            SELECT * FROM bets 
            ORDER BY date DESC, timestamp DESC
        """
        
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(filename, index=False)
        
        logger.info(f"Data exported to {filename}")
        return filename
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

class AutomatedBettingSystem:
    """Automated system that finds opportunities and tracks bets."""
    
    def __init__(self, tracker: BetTracker, live_system: LivePredictionSystem,
                 auto_bet: bool = False, min_confidence: float = 6.0):
        self.tracker = tracker
        self.live_system = live_system
        self.auto_bet = auto_bet  # Set to True for actual betting
        self.min_confidence = min_confidence
        
    def run_daily_analysis(self, date: str = None) -> Dict:
        """Run daily analysis and place bets."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Running daily analysis for {date}")
        
        try:
            # Get betting opportunities
            opportunities = self.live_system.get_todays_betting_opportunities(date)
            
            if not opportunities:
                logger.info("No betting opportunities found")
                return {"opportunities": 0, "bets_placed": 0}
            
            # Filter by confidence threshold
            high_confidence_opps = [
                opp for opp in opportunities 
                if opp.confidence_score >= self.min_confidence
            ]
            
            logger.info(f"Found {len(opportunities)} opportunities, {len(high_confidence_opps)} above confidence threshold")
            
            bets_placed = 0
            
            for opp in high_confidence_opps:
                # Calculate bet size (conservative Kelly)
                bet_amount = min(
                    opp.kelly_fraction * self.tracker.bankroll * 0.5,  # 50% of Kelly for safety
                    50.0  # Max $50 per bet
                )
                
                # Skip very small bets
                if bet_amount < 5.0:
                    continue
                
                # Create bet record
                bet_record = BetRecord(
                    bet_id="",  # Will be auto-generated
                    date=date,
                    player_name=opp.player_name,
                    model_probability=opp.model_probability,
                    odds=opp.odds_yes,
                    bet_amount=bet_amount,
                    expected_value=opp.expected_value,
                    kelly_fraction=opp.kelly_fraction,
                    confidence_score=opp.confidence_score,
                    bookmaker=opp.bookmaker,
                    notes=f"Auto-generated bet from daily analysis"
                )
                
                # Place bet (record in database)
                if self.tracker.place_bet(bet_record):
                    bets_placed += 1
                    
                    if self.auto_bet:
                        # TODO: Integrate with actual sportsbook API
                        logger.info(f"Would place real bet: {bet_record.player_name} ${bet_amount:.2f}")
                    else:
                        logger.info(f"Paper bet placed: {bet_record.player_name} ${bet_amount:.2f}")
            
            return {
                "date": date,
                "opportunities": len(opportunities),
                "high_confidence": len(high_confidence_opps),
                "bets_placed": bets_placed
            }
            
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")
            return {"error": str(e)}
    
    def update_results_from_api(self, date: str = None):
        """Update bet results by checking game outcomes via API."""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Get pending bets for the date
        pending_bets = self.tracker.conn.execute("""
            SELECT bet_id, player_name, game_id 
            FROM bets 
            WHERE date = ? AND status = 'pending'
        """, (date,)).fetchall()
        
        if not pending_bets:
            logger.info(f"No pending bets for {date}")
            return
        
        logger.info(f"Checking results for {len(pending_bets)} bets on {date}")
        
        # TODO: Implement actual API calls to check game results
        # For now, simulate random results for testing
        for bet_id, player_name, game_id in pending_bets:
            # Simulate 15% win rate (realistic for home run bets)
            won = np.random.random() < 0.15
            self.tracker.update_bet_result(bet_id, won)
            
            logger.info(f"Updated {player_name}: {'WON' if won else 'LOST'}")

def create_weekly_tracking_script():
    """Create the main script for weekly tracking."""
    
    script_content = '''#!/usr/bin/env python3
"""
Weekly Bet Tracking Script
=========================

Run this daily to track your betting performance.
"""

import logging
from datetime import datetime
from bet_tracker import BetTracker, AutomatedBettingSystem
from live_prediction_system import LivePredictionSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'betting_log_{datetime.now().strftime("%Y%m")}.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Main execution function."""
    
    # Initialize systems
    tracker = BetTracker(
        db_path="betting_tracker.db",
        max_daily_risk=100.0,  # Max $100 per day
        bankroll=1000.0        # Starting bankroll
    )
    
    live_system = LivePredictionSystem()
    live_system.initialize()
    
    auto_system = AutomatedBettingSystem(
        tracker=tracker,
        live_system=live_system,
        auto_bet=False,  # Set to True when ready for real betting
        min_confidence=6.0
    )
    
    try:
        # Run today's analysis
        today = datetime.now().strftime("%Y-%m-%d")
        results = auto_system.run_daily_analysis(today)
        
        print(f"\\nDaily Analysis Results for {today}:")
        print(f"  Opportunities found: {results.get('opportunities', 0)}")
        print(f"  High confidence: {results.get('high_confidence', 0)}")
        print(f"  Bets placed: {results.get('bets_placed', 0)}")
        
        # Update yesterday's results
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        auto_system.update_results_from_api(yesterday)
        
        # Generate performance report
        report = tracker.generate_performance_report(days=7)
        print(report)
        
        # Export data
        csv_file = tracker.export_to_csv()
        print(f"\\nData exported to: {csv_file}")
        
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
        
    finally:
        tracker.close()

if __name__ == "__main__":
    main()
'''
    
    with open("weekly_tracker.py", "w") as f:
        f.write(script_content)
    
    print("Created weekly_tracker.py")

if __name__ == "__main__":
    # Demo usage
    print("Setting up bet tracking system...")
    
    # Create the weekly tracking script
    create_weekly_tracking_script()
    
    # Demo the system
    tracker = BetTracker()
    
    # Example bet
    example_bet = BetRecord(
        bet_id="",
        date="2025-08-16",
        player_name="Mike Trout",
        model_probability=0.18,
        odds=450,
        bet_amount=25.0,
        expected_value=0.125,
        kelly_fraction=0.08,
        confidence_score=7.2,
        bookmaker="DraftKings"
    )
    
    tracker.place_bet(example_bet)
    print("Example bet placed!")
    
    # Show report
    report = tracker.generate_performance_report(days=30)
    print(report)
    
    tracker.close()