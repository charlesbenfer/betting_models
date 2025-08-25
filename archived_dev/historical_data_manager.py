"""
Historical Data Management System
=================================

Manages a centralized database of historical MLB data for prediction generation.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging

from config import config

logger = logging.getLogger(__name__)

class HistoricalDataManager:
    """
    Manages centralized historical MLB data storage and retrieval.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(config.DATA_DIR) / "historical_mlb_data.db")
        self.conn = None
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            
            # Create main games table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    game_date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    venue TEXT,
                    game_time TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create player performances table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS player_performances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    position TEXT,
                    at_bats INTEGER DEFAULT 0,
                    hits INTEGER DEFAULT 0,
                    home_runs INTEGER DEFAULT 0,
                    rbis INTEGER DEFAULT 0,
                    walks INTEGER DEFAULT 0,
                    strikeouts INTEGER DEFAULT 0,
                    batting_avg REAL,
                    on_base_pct REAL,
                    slugging_pct REAL,
                    woba REAL,
                    wrc_plus REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
            """)
            
            # Create pitching matchups table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pitching_matchups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    pitcher_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    is_starter BOOLEAN DEFAULT TRUE,
                    innings_pitched REAL DEFAULT 0,
                    hits_allowed INTEGER DEFAULT 0,
                    home_runs_allowed INTEGER DEFAULT 0,
                    walks_allowed INTEGER DEFAULT 0,
                    strikeouts INTEGER DEFAULT 0,
                    earned_runs INTEGER DEFAULT 0,
                    era REAL,
                    whip REAL,
                    k9 REAL,
                    bb9 REAL,
                    hr9 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
            """)
            
            # Create weather conditions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction TEXT,
                    weather_condition TEXT,
                    barometric_pressure REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
            """)
            
            # Create team stats table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS team_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    runs_scored INTEGER DEFAULT 0,
                    runs_allowed INTEGER DEFAULT 0,
                    home_runs_hit INTEGER DEFAULT 0,
                    home_runs_allowed INTEGER DEFAULT 0,
                    batting_avg REAL,
                    on_base_pct REAL,
                    slugging_pct REAL,
                    era REAL,
                    whip REAL,
                    recent_form TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games (game_date)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_player_perf_game ON player_performances (game_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_player_perf_name ON player_performances (player_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pitching_game ON pitching_matchups (game_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_team_stats_date ON team_stats (game_date)")
            
            self.conn.commit()
            logger.info(f"Historical database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def load_existing_data(self):
        """Load all existing parquet files into the database."""
        try:
            data_dir = Path(config.DATA_DIR) / "processed"
            
            if not data_dir.exists():
                logger.warning(f"Processed data directory not found: {data_dir}")
                return
            
            # Find all pregame dataset files
            pregame_files = list(data_dir.glob("pregame_dataset_*.parquet"))
            logger.info(f"Found {len(pregame_files)} pregame dataset files")
            
            total_games_added = 0
            total_performances_added = 0
            
            for file_path in pregame_files:
                logger.info(f"Processing {file_path.name}")
                
                try:
                    df = pd.read_parquet(file_path)
                    games_added, performances_added = self._import_pregame_data(df)
                    total_games_added += games_added
                    total_performances_added += performances_added
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    continue
            
            logger.info(f"Data import complete: {total_games_added} games, {total_performances_added} performances")
            
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
    
    def _import_pregame_data(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Import pregame dataset into database."""
        games_added = 0
        performances_added = 0
        
        try:
            # Process each row as a game/player performance
            for _, row in df.iterrows():
                try:
                    # Create game record
                    game_date = str(row.get('date', ''))
                    home_team = str(row.get('home_team', ''))
                    away_team = str(row.get('away_team', ''))
                    
                    if not all([game_date, home_team, away_team]):
                        continue
                    
                    game_id = f"{game_date}_{home_team}_{away_team}"
                    
                    # Insert game
                    cursor = self.conn.execute("""
                        INSERT OR IGNORE INTO games 
                        (game_id, game_date, home_team, away_team, venue)
                        VALUES (?, ?, ?, ?, ?)
                    """, (game_id, game_date, home_team, away_team, row.get('venue', '')))
                    
                    if cursor.rowcount > 0:
                        games_added += 1
                    
                    # Insert player performance
                    player_name = str(row.get('batter_name', ''))
                    if player_name and player_name != 'nan':
                        cursor = self.conn.execute("""
                            INSERT OR IGNORE INTO player_performances
                            (game_id, player_name, team, at_bats, hits, home_runs, 
                             batting_avg, on_base_pct, slugging_pct, woba)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            game_id, player_name, str(row.get('batter_team', '')),
                            row.get('recent_at_bats', 0), row.get('recent_hits', 0), 
                            row.get('recent_home_runs', 0), row.get('recent_batting_avg', 0),
                            row.get('recent_on_base_pct', 0), row.get('recent_slugging_pct', 0),
                            row.get('recent_woba', 0)
                        ))
                        
                        if cursor.rowcount > 0:
                            performances_added += 1
                    
                    # Insert pitcher data
                    pitcher_name = str(row.get('pitcher_name', ''))
                    if pitcher_name and pitcher_name != 'nan':
                        self.conn.execute("""
                            INSERT OR IGNORE INTO pitching_matchups
                            (game_id, pitcher_name, team, era, whip, k9, bb9, hr9)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            game_id, pitcher_name, str(row.get('pitcher_team', '')),
                            row.get('pitcher_era', 0), row.get('pitcher_whip', 0),
                            row.get('pitcher_k9', 0), row.get('pitcher_bb9', 0), 
                            row.get('pitcher_hr9', 0)
                        ))
                    
                    # Insert weather data
                    if any(row.get(col) for col in ['temperature', 'humidity', 'wind_speed']):
                        self.conn.execute("""
                            INSERT OR IGNORE INTO weather_conditions
                            (game_id, temperature, humidity, wind_speed, wind_direction)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            game_id, row.get('temperature'), row.get('humidity'),
                            row.get('wind_speed'), row.get('wind_direction', '')
                        ))
                
                except Exception as e:
                    logger.warning(f"Failed to import row: {e}")
                    continue
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to import pregame data: {e}")
            self.conn.rollback()
        
        return games_added, performances_added
    
    def get_recent_data(self, days: int = 45, end_date: str = None) -> pd.DataFrame:
        """Get recent data for predictions."""
        try:
            if end_date is None:
                # Use the most recent date in our database
                cursor = self.conn.execute("SELECT MAX(game_date) FROM games")
                result = cursor.fetchone()
                if result[0]:
                    end_date = result[0]
                else:
                    logger.warning("No data found in database")
                    return pd.DataFrame()
            
            # Calculate start date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=days)
            start_date = start_dt.strftime("%Y-%m-%d")
            
            logger.info(f"Retrieving data from {start_date} to {end_date}")
            
            # Complex query to rebuild pregame-style data
            query = """
                SELECT 
                    g.game_date,
                    g.home_team,
                    g.away_team,
                    g.venue,
                    pp.player_name as batter_name,
                    pp.team as batter_team,
                    pp.at_bats as recent_at_bats,
                    pp.hits as recent_hits,
                    pp.home_runs as recent_home_runs,
                    pp.batting_avg as recent_batting_avg,
                    pp.on_base_pct as recent_on_base_pct,
                    pp.slugging_pct as recent_slugging_pct,
                    pp.woba as recent_woba,
                    pm.pitcher_name,
                    pm.team as pitcher_team,
                    pm.era as pitcher_era,
                    pm.whip as pitcher_whip,
                    pm.k9 as pitcher_k9,
                    pm.bb9 as pitcher_bb9,
                    pm.hr9 as pitcher_hr9,
                    wc.temperature,
                    wc.humidity,
                    wc.wind_speed,
                    wc.wind_direction
                FROM games g
                LEFT JOIN player_performances pp ON g.game_id = pp.game_id
                LEFT JOIN pitching_matchups pm ON g.game_id = pm.game_id AND pm.is_starter = 1
                LEFT JOIN weather_conditions wc ON g.game_id = wc.game_id
                WHERE g.game_date BETWEEN ? AND ?
                  AND pp.player_name IS NOT NULL
                ORDER BY g.game_date DESC, g.game_id
            """
            
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            
            logger.info(f"Retrieved {len(df)} records for prediction features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return pd.DataFrame()
    
    def append_daily_data(self, new_data: pd.DataFrame):
        """Append new daily data to the database."""
        try:
            games_added, performances_added = self._import_pregame_data(new_data)
            logger.info(f"Appended daily data: {games_added} games, {performances_added} performances")
            
        except Exception as e:
            logger.error(f"Failed to append daily data: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Game count
            cursor = self.conn.execute("SELECT COUNT(*) FROM games")
            stats['total_games'] = cursor.fetchone()[0]
            
            # Player performance count
            cursor = self.conn.execute("SELECT COUNT(*) FROM player_performances")
            stats['total_performances'] = cursor.fetchone()[0]
            
            # Date range
            cursor = self.conn.execute("SELECT MIN(game_date), MAX(game_date) FROM games")
            result = cursor.fetchone()
            stats['date_range'] = {'start': result[0], 'end': result[1]}
            
            # Unique players
            cursor = self.conn.execute("SELECT COUNT(DISTINCT player_name) FROM player_performances")
            stats['unique_players'] = cursor.fetchone()[0]
            
            # Unique teams
            cursor = self.conn.execute("SELECT COUNT(DISTINCT home_team) FROM games")
            stats['unique_teams'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def initialize_historical_database():
    """Initialize the historical database with existing data."""
    try:
        print("🏗️  Initializing Historical Data Database...")
        
        with HistoricalDataManager() as hdm:
            print("📊 Loading existing parquet files...")
            hdm.load_existing_data()
            
            stats = hdm.get_database_stats()
            print("\n✅ Database Statistics:")
            print(f"   Total Games: {stats.get('total_games', 0):,}")
            print(f"   Player Performances: {stats.get('total_performances', 0):,}")
            print(f"   Unique Players: {stats.get('unique_players', 0):,}")
            print(f"   Unique Teams: {stats.get('unique_teams', 0):,}")
            
            if stats.get('date_range'):
                print(f"   Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            
            print("\n🎯 Historical database ready!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False


if __name__ == "__main__":
    initialize_historical_database()