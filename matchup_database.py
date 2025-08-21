"""
Production Matchup Database System
=================================

A production-ready system for storing and retrieving batter-pitcher matchup data
for fast inference without rebuilding history from scratch.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from data_utils import DataValidator, CacheManager

logger = logging.getLogger(__name__)

@dataclass
class MatchupRecord:
    """Represents a batter-pitcher matchup record."""
    batter_id: int
    pitcher_id: int
    career_pa: int
    career_hr: int
    career_hr_rate: float
    career_avg_ev: float
    career_avg_la: float
    recent_pa: int  # Last 5 encounters
    recent_hr: int
    recent_hr_rate: float
    last_encounter_date: Optional[str]
    encounters_last_year: int
    familiarity_score: float
    last_updated: str

@dataclass
class PitcherProfile:
    """Represents a pitcher's profile for similarity matching."""
    pitcher_id: int
    avg_velocity: float
    handedness: str
    primary_pitch: str
    velocity_tier: str
    last_updated: str

class MatchupDatabase:
    """Production-ready matchup database with fast lookups."""
    
    def __init__(self, db_path: str = "data/matchup_database.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.validator = DataValidator()
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Batter-pitcher matchup table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matchups (
                    batter_id INTEGER,
                    pitcher_id INTEGER,
                    career_pa INTEGER,
                    career_hr INTEGER,
                    career_hr_rate REAL,
                    career_avg_ev REAL,
                    career_avg_la REAL,
                    recent_pa INTEGER,
                    recent_hr INTEGER,
                    recent_hr_rate REAL,
                    last_encounter_date TEXT,
                    encounters_last_year INTEGER,
                    familiarity_score REAL,
                    last_updated TEXT,
                    PRIMARY KEY (batter_id, pitcher_id)
                )
            """)
            
            # Pitcher profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pitcher_profiles (
                    pitcher_id INTEGER PRIMARY KEY,
                    avg_velocity REAL,
                    handedness TEXT,
                    primary_pitch TEXT,
                    velocity_tier TEXT,
                    last_updated TEXT
                )
            """)
            
            # Similarity groups cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS similarity_stats (
                    batter_id INTEGER,
                    group_type TEXT,  -- 'same_handedness' or 'same_velocity_tier'
                    group_value TEXT, -- 'R'/'L' for handedness, 'high'/'medium'/'low' for velocity
                    pa INTEGER,
                    hr INTEGER,
                    hr_rate REAL,
                    last_updated TEXT,
                    PRIMARY KEY (batter_id, group_type, group_value)
                )
            """)
            
            # Metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    last_updated TEXT
                )
            """)
            
            # Create indexes for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_matchups_batter ON matchups(batter_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_matchups_pitcher ON matchups(pitcher_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_similarity_batter ON similarity_stats(batter_id)")
            
            conn.commit()
        
        logger.info(f"Initialized matchup database at {self.db_path}")
    
    def get_matchup_features(self, batter_id: int, pitcher_id: int) -> Dict[str, float]:
        """Fast lookup of matchup features for inference."""
        with sqlite3.connect(self.db_path) as conn:
            # Get direct matchup
            matchup_query = """
                SELECT career_pa, career_hr, career_hr_rate, career_avg_ev, career_avg_la,
                       recent_pa, recent_hr, recent_hr_rate, last_encounter_date,
                       encounters_last_year, familiarity_score
                FROM matchups 
                WHERE batter_id = ? AND pitcher_id = ?
            """
            
            result = conn.execute(matchup_query, (batter_id, pitcher_id)).fetchone()
            
            if result:
                # We have direct matchup history
                matchup_features = {
                    'matchup_pa_career': float(result[0]),
                    'matchup_hr_career': float(result[1]),
                    'matchup_hr_rate_career': float(result[2]),
                    'matchup_avg_ev_career': float(result[3] or 0),
                    'matchup_avg_la_career': float(result[4] or 0),
                    'matchup_pa_recent': float(result[5]),
                    'matchup_hr_recent': float(result[6]),
                    'matchup_hr_rate_recent': float(result[7]),
                    'matchup_days_since_last': self._calculate_days_since(result[8]),
                    'matchup_encounters_last_year': float(result[9]),
                    'matchup_familiarity_score': float(result[10])
                }
            else:
                # No direct history - use defaults
                matchup_features = self._get_default_matchup_features()
            
            # Get similarity features
            similarity_features = self._get_similarity_features_fast(batter_id, pitcher_id, conn)
            
            # Combine all features
            return {**matchup_features, **similarity_features}
    
    def _get_similarity_features_fast(self, batter_id: int, pitcher_id: int, conn) -> Dict[str, float]:
        """Fast lookup of similarity features."""
        # Get pitcher profile
        pitcher_query = "SELECT handedness, velocity_tier FROM pitcher_profiles WHERE pitcher_id = ?"
        pitcher_result = conn.execute(pitcher_query, (pitcher_id,)).fetchone()
        
        if not pitcher_result:
            return self._get_default_similarity_features()
        
        handedness, velocity_tier = pitcher_result
        
        # Get similarity stats for this batter
        similarity_query = """
            SELECT group_type, group_value, pa, hr, hr_rate 
            FROM similarity_stats 
            WHERE batter_id = ? AND 
                  ((group_type = 'same_handedness' AND group_value = ?) OR
                   (group_type = 'same_velocity_tier' AND group_value = ?))
        """
        
        similarity_results = conn.execute(
            similarity_query, 
            (batter_id, handedness, velocity_tier)
        ).fetchall()
        
        # Initialize defaults
        features = self._get_default_similarity_features()
        
        # Update with actual data
        for group_type, group_value, pa, hr, hr_rate in similarity_results:
            if group_type == 'same_handedness':
                features['vs_similar_hand_pa'] = float(pa)
                features['vs_similar_hand_hr'] = float(hr)
                features['vs_similar_hand_hr_rate'] = float(hr_rate)
            elif group_type == 'same_velocity_tier':
                features['vs_similar_velocity_pa'] = float(pa)
                features['vs_similar_velocity_hr'] = float(hr)
                features['vs_similar_velocity_hr_rate'] = float(hr_rate)
        
        return features
    
    def bulk_update_matchups(self, statcast_df: pd.DataFrame, 
                           cutoff_date: Optional[str] = None) -> Dict[str, int]:
        """Incrementally update matchup database with new Statcast data."""
        logger.info("Starting bulk matchup database update...")
        
        if cutoff_date:
            cutoff_date = pd.to_datetime(cutoff_date)
            statcast_df = statcast_df[statcast_df['date'] >= cutoff_date].copy()
        
        logger.info(f"Processing {len(statcast_df)} Statcast records for matchup updates")
        
        # Build updated matchup records
        updated_matchups = self._build_matchup_records(statcast_df)
        updated_profiles = self._build_pitcher_profiles(statcast_df)
        updated_similarity = self._build_similarity_stats(statcast_df)
        
        # Update database
        stats = self._write_updates_to_db(updated_matchups, updated_profiles, updated_similarity)
        
        # Update metadata
        self._update_metadata("last_full_update", datetime.now().isoformat())
        
        logger.info(f"Matchup database update complete: {stats}")
        return stats
    
    def _build_matchup_records(self, statcast_df: pd.DataFrame) -> List[MatchupRecord]:
        """Build matchup records from Statcast data."""
        # Group by batter-pitcher pairs
        matchup_groups = statcast_df.groupby(['batter', 'pitcher'])
        
        records = []
        for (batter_id, pitcher_id), group in matchup_groups:
            group = group.sort_values('date')
            
            # Career stats
            career_pa = group['is_pa'].sum()
            career_hr = group['is_hr'].sum()
            career_hr_rate = career_hr / max(1, career_pa)
            career_avg_ev = group['launch_speed'].mean() if group['launch_speed'].notna().sum() > 0 else 0
            career_avg_la = group['launch_angle'].mean() if group['launch_angle'].notna().sum() > 0 else 0
            
            # Recent stats (last 5 encounters)
            recent_group = group.tail(5)
            recent_pa = recent_group['is_pa'].sum()
            recent_hr = recent_group['is_hr'].sum()
            recent_hr_rate = recent_hr / max(1, recent_pa)
            
            # Timing stats
            last_encounter = group['date'].max().isoformat() if len(group) > 0 else None
            one_year_ago = datetime.now() - timedelta(days=365)
            recent_encounters = group[group['date'] >= one_year_ago]
            encounters_last_year = len(recent_encounters)
            
            # Familiarity score
            familiarity_score = min(career_pa / 20, 1.0)
            
            record = MatchupRecord(
                batter_id=int(batter_id),
                pitcher_id=int(pitcher_id),
                career_pa=int(career_pa),
                career_hr=int(career_hr),
                career_hr_rate=float(career_hr_rate),
                career_avg_ev=float(career_avg_ev),
                career_avg_la=float(career_avg_la),
                recent_pa=int(recent_pa),
                recent_hr=int(recent_hr),
                recent_hr_rate=float(recent_hr_rate),
                last_encounter_date=last_encounter,
                encounters_last_year=int(encounters_last_year),
                familiarity_score=float(familiarity_score),
                last_updated=datetime.now().isoformat()
            )
            records.append(record)
        
        return records
    
    def _build_pitcher_profiles(self, statcast_df: pd.DataFrame) -> List[PitcherProfile]:
        """Build pitcher profiles from Statcast data."""
        pitcher_groups = statcast_df.groupby('pitcher')
        
        profiles = []
        for pitcher_id, group in pitcher_groups:
            avg_velocity = group['release_speed'].mean() if group['release_speed'].notna().sum() > 0 else 92.0
            handedness = group['p_throws'].mode().iloc[0] if len(group['p_throws'].mode()) > 0 else 'R'
            primary_pitch = group['pitch_type'].mode().iloc[0] if len(group['pitch_type'].mode()) > 0 else 'FF'
            
            # Velocity tier
            if avg_velocity > 94:
                velocity_tier = 'high'
            elif avg_velocity > 90:
                velocity_tier = 'medium'
            else:
                velocity_tier = 'low'
            
            profile = PitcherProfile(
                pitcher_id=int(pitcher_id),
                avg_velocity=float(avg_velocity),
                handedness=str(handedness),
                primary_pitch=str(primary_pitch),
                velocity_tier=str(velocity_tier),
                last_updated=datetime.now().isoformat()
            )
            profiles.append(profile)
        
        return profiles
    
    def _build_similarity_stats(self, statcast_df: pd.DataFrame) -> List[Dict]:
        """Build similarity statistics from Statcast data."""
        # This is a simplified version - in production, we'd want more sophisticated similarity calculation
        similarity_stats = []
        
        # Get pitcher profiles for grouping
        pitcher_profiles = {p.pitcher_id: p for p in self._build_pitcher_profiles(statcast_df)}
        
        # Group batters by their performance vs pitcher types
        batter_groups = statcast_df.groupby('batter')
        
        for batter_id, batter_data in batter_groups:
            # Performance vs each handedness
            for handedness in ['R', 'L']:
                hand_data = batter_data[batter_data['pitcher'].apply(
                    lambda pid: pitcher_profiles.get(pid, type('obj', (object,), {'handedness': 'R'})).handedness == handedness
                )]
                
                if len(hand_data) > 0:
                    pa = hand_data['is_pa'].sum()
                    hr = hand_data['is_hr'].sum()
                    hr_rate = hr / max(1, pa)
                    
                    similarity_stats.append({
                        'batter_id': int(batter_id),
                        'group_type': 'same_handedness',
                        'group_value': handedness,
                        'pa': int(pa),
                        'hr': int(hr),
                        'hr_rate': float(hr_rate),
                        'last_updated': datetime.now().isoformat()
                    })
            
            # Performance vs each velocity tier
            for velocity_tier in ['high', 'medium', 'low']:
                tier_data = batter_data[batter_data['pitcher'].apply(
                    lambda pid: pitcher_profiles.get(pid, type('obj', (object,), {'velocity_tier': 'medium'})).velocity_tier == velocity_tier
                )]
                
                if len(tier_data) > 0:
                    pa = tier_data['is_pa'].sum()
                    hr = tier_data['is_hr'].sum()
                    hr_rate = hr / max(1, pa)
                    
                    similarity_stats.append({
                        'batter_id': int(batter_id),
                        'group_type': 'same_velocity_tier',
                        'group_value': velocity_tier,
                        'pa': int(pa),
                        'hr': int(hr),
                        'hr_rate': float(hr_rate),
                        'last_updated': datetime.now().isoformat()
                    })
        
        return similarity_stats
    
    def _write_updates_to_db(self, matchups: List[MatchupRecord], 
                           profiles: List[PitcherProfile],
                           similarity_stats: List[Dict]) -> Dict[str, int]:
        """Write updates to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Update matchups
            matchup_data = [tuple(asdict(m).values()) for m in matchups]
            conn.executemany("""
                INSERT OR REPLACE INTO matchups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, matchup_data)
            
            # Update pitcher profiles
            profile_data = [tuple(asdict(p).values()) for p in profiles]
            conn.executemany("""
                INSERT OR REPLACE INTO pitcher_profiles VALUES (?,?,?,?,?,?)
            """, profile_data)
            
            # Update similarity stats
            similarity_data = [tuple(s.values()) for s in similarity_stats]
            conn.executemany("""
                INSERT OR REPLACE INTO similarity_stats VALUES (?,?,?,?,?,?,?)
            """, similarity_data)
            
            conn.commit()
        
        return {
            'matchups_updated': len(matchups),
            'profiles_updated': len(profiles),
            'similarity_stats_updated': len(similarity_stats)
        }
    
    def _calculate_days_since(self, last_encounter_date: Optional[str]) -> float:
        """Calculate days since last encounter."""
        if not last_encounter_date:
            return 999.0
        
        try:
            last_date = pd.to_datetime(last_encounter_date)
            days_since = (datetime.now() - last_date).days
            return float(min(days_since, 999))
        except:
            return 999.0
    
    def _get_default_matchup_features(self) -> Dict[str, float]:
        """Default matchup features when no history exists."""
        return {
            'matchup_pa_career': 0.0,
            'matchup_hr_career': 0.0,
            'matchup_hr_rate_career': 0.0,
            'matchup_avg_ev_career': 0.0,
            'matchup_avg_la_career': 0.0,
            'matchup_pa_recent': 0.0,
            'matchup_hr_recent': 0.0,
            'matchup_hr_rate_recent': 0.0,
            'matchup_days_since_last': 999.0,
            'matchup_encounters_last_year': 0.0,
            'matchup_familiarity_score': 0.0
        }
    
    def _get_default_similarity_features(self) -> Dict[str, float]:
        """Default similarity features when no data exists."""
        return {
            'vs_similar_hand_pa': 0.0,
            'vs_similar_hand_hr': 0.0,
            'vs_similar_hand_hr_rate': 0.0,
            'vs_similar_velocity_pa': 0.0,
            'vs_similar_velocity_hr': 0.0,
            'vs_similar_velocity_hr_rate': 0.0
        }
    
    def _update_metadata(self, key: str, value: str):
        """Update metadata table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metadata VALUES (?, ?, ?)
            """, (key, value, datetime.now().isoformat()))
            conn.commit()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            matchup_count = conn.execute("SELECT COUNT(*) FROM matchups").fetchone()[0]
            profile_count = conn.execute("SELECT COUNT(*) FROM pitcher_profiles").fetchone()[0]
            similarity_count = conn.execute("SELECT COUNT(*) FROM similarity_stats").fetchone()[0]
            
            last_update = conn.execute(
                "SELECT value FROM metadata WHERE key = 'last_full_update'"
            ).fetchone()
            
            return {
                'matchup_records': matchup_count,
                'pitcher_profiles': profile_count,
                'similarity_records': similarity_count,
                'last_update': last_update[0] if last_update else None,
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024)
            }

# Export the main class
__all__ = ['MatchupDatabase', 'MatchupRecord', 'PitcherProfile']