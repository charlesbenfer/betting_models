"""
API Client Module
================

Handles external API calls for odds data and MLB statistics.
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import json

from data_utils import DataValidator

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # Check if we need to wait
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        self.call_times.append(now)

class TheOddsAPIClient:
    """Client for The Odds API."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com/v4"):
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(calls_per_minute=50)  # Conservative limit
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Baseball-HR-Predictor/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Dict[str, Any], 
                     timeout: int = 30, max_retries: int = 3) -> Dict[str, Any]:
        """Make a request to the API with retry logic."""
        params['apiKey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    raise APIError(f"All {max_retries} API attempts failed: {e}")
                
                # Exponential backoff
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        
        raise APIError("Unexpected error in API request")
    
    def get_mlb_events(self, date_filter: Optional[str] = None, 
                       regions: str = "us,us2") -> List[Dict[str, Any]]:
        """Get MLB events for a specific date."""
        params = {
            'regions': regions,
            'dateFormat': 'iso'
        }
        
        if date_filter:
            params['date'] = date_filter
        
        try:
            data = self._make_request("sports/baseball_mlb/events", params)
            logger.info(f"Fetched {len(data)} MLB events")
            return data
        except APIError as e:
            logger.error(f"Failed to fetch MLB events: {e}")
            return []
    
    def get_event_odds(self, event_id: str, market: str = "batter_home_runs",
                       regions: str = "us,us2", bookmakers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get odds for a specific event."""
        params = {
            'regions': regions,
            'markets': market,
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        try:
            endpoint = f"sports/baseball_mlb/events/{event_id}/odds"
            return self._make_request(endpoint, params)
        except APIError as e:
            logger.warning(f"Failed to fetch odds for event {event_id}: {e}")
            return {}
    
    def get_home_run_odds_for_date(self, target_date: str, 
                                  timezone: str = "America/Chicago") -> pd.DataFrame:
        """Get home run odds for a specific date."""
        logger.info(f"Fetching home run odds for {target_date}")
        
        # Convert target date
        target_dt = pd.to_datetime(target_date).tz_localize(timezone).normalize()
        
        # Get events for the date
        events = self.get_mlb_events()
        
        # Filter events for target date
        events_today = []
        for event in events:
            if 'commence_time' in event:
                event_time = pd.to_datetime(event['commence_time'], utc=True)
                event_local = event_time.tz_convert(timezone).normalize()
                
                if event_local == target_dt:
                    events_today.append(event)
        
        logger.info(f"Found {len(events_today)} games for {target_date}")
        
        if not events_today:
            return pd.DataFrame()
        
        # Collect odds for each event
        all_odds = {}  # {(bookmaker, player): {"over": price, "under": price}}
        
        for event in events_today:
            event_id = event.get('id')
            if not event_id:
                continue
            
            odds_data = self.get_event_odds(event_id)
            
            if isinstance(odds_data, dict):
                bookmakers = odds_data.get('bookmakers', [])
            elif isinstance(odds_data, list):
                bookmakers = odds_data
            else:
                continue
            
            for bookmaker in bookmakers:
                if not isinstance(bookmaker, dict):
                    continue
                
                bk_key = (bookmaker.get('key') or 
                         bookmaker.get('bookmaker_key') or 
                         bookmaker.get('bookmaker', ''))
                
                markets = bookmaker.get('markets', [])
                
                for market in markets:
                    if not isinstance(market, dict) or market.get('key') != 'batter_home_runs':
                        continue
                    
                    outcomes = market.get('outcomes', [])
                    
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue
                        
                        player_name = (outcome.get('description') or 
                                     outcome.get('participant') or 
                                     outcome.get('name', '')).strip()
                        
                        side = (outcome.get('name', '')).strip().lower()
                        price = outcome.get('price')
                        point = outcome.get('point')
                        
                        if not player_name or price is None or point is None:
                            continue
                        
                        # Only process Over/Under 0.5 HR markets
                        try:
                            if abs(float(point) - 0.5) > 1e-6:
                                continue
                        except (ValueError, TypeError):
                            continue
                        
                        # Store odds
                        key = (bk_key, DataValidator.standardize_name(player_name))
                        if key not in all_odds:
                            all_odds[key] = {}
                        
                        if side == 'over':
                            all_odds[key]['over'] = price
                        elif side == 'under':
                            all_odds[key]['under'] = price
        
        # Convert to DataFrame
        rows = []
        
        # Group by player and select best odds
        by_player = {}
        for (bookmaker, player), odds in all_odds.items():
            if 'over' not in odds:
                continue
            
            current_best = by_player.get(player)
            if current_best is None or odds['over'] > current_best['over']:
                by_player[player] = {
                    'bookmaker': bookmaker,
                    'over': odds['over'],
                    'under': odds.get('under')
                }
        
        # Create final rows
        for player, best_odds in by_player.items():
            over_odds = best_odds['over']
            under_odds = best_odds['under']
            
            # Calculate no-vig probability if both sides available
            p_novig = np.nan
            if under_odds is not None:
                p_novig = self._calculate_no_vig_probability(over_odds, under_odds)
            
            rows.append({
                'date': target_dt.tz_convert(None),
                'batter_name': player,
                'odds_hr_yes': over_odds,
                'odds_hr_no': under_odds if under_odds is not None else np.nan,
                'p_book_novig': p_novig,
                'bookmaker': best_odds['bookmaker']
            })
        
        if not rows:
            logger.warning("No home run odds found for the target date")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        
        logger.info(f"Collected odds for {len(df)} players")
        return df
    
    @staticmethod
    def _calculate_no_vig_probability(yes_odds: float, no_odds: float) -> float:
        """Calculate no-vig probability from two-way odds."""
        try:
            # Convert American odds to implied probabilities
            if yes_odds >= 0:
                p_yes = 100 / (yes_odds + 100)
            else:
                p_yes = -yes_odds / (-yes_odds + 100)
            
            if no_odds >= 0:
                p_no = 100 / (no_odds + 100)
            else:
                p_no = -no_odds / (-no_odds + 100)
            
            # Remove vig by normalizing
            total = p_yes + p_no
            if total > 0:
                return p_yes / total
            
        except (ValueError, ZeroDivisionError):
            pass
        
        return np.nan

class MLBStatsAPIClient:
    """Client for MLB Stats API."""
    
    def __init__(self, base_url: str = "https://statsapi.mlb.com/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(calls_per_minute=100)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None,
                     timeout: int = 20, max_retries: int = 3) -> Dict[str, Any]:
        """Make a request to MLB Stats API with retry logic."""
        if params is None:
            params = {}
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"MLB API request attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    raise APIError(f"All {max_retries} MLB API attempts failed: {e}")
                
                # Exponential backoff
                sleep_time = 1.5 ** (attempt + 1)
                time.sleep(sleep_time)
        
        raise APIError("Unexpected error in MLB API request")
    
    def get_probable_starters(self, date: str, timezone: str = "America/Chicago") -> pd.DataFrame:
        """Get probable starting pitchers for a given date."""
        try:
            date_obj = pd.to_datetime(date).tz_localize(timezone).normalize()
            date_str = date_obj.strftime('%Y-%m-%d')
            
            params = {
                'sportId': 1,
                'date': date_str,
                'hydrate': 'probablePitcher'
            }
            
            data = self._make_request("schedule", params)
            
            rows = []
            for date_entry in data.get("dates", []):
                for game in date_entry.get("games", []):
                    game_pk = game.get("gamePk")
                    teams = game.get("teams", {})
                    
                    for side in ("home", "away"):
                        side_data = teams.get(side, {})
                        probable_pitcher = side_data.get("probablePitcher", {})
                        
                        if probable_pitcher.get("id"):
                            rows.append({
                                "game_pk": game_pk,
                                "date": date_obj.tz_convert(None),
                                "side": side,
                                "starter_pitcher": int(probable_pitcher["id"]),
                                "starter_name": probable_pitcher.get("fullName", ""),
                            })
            
            df = pd.DataFrame(rows)
            logger.info(f"Found {len(df)} probable starters for {date_str}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get probable starters: {e}")
            return pd.DataFrame(columns=['game_pk', 'date', 'side', 'starter_pitcher', 'starter_name'])

class SafeAPIClient:
    """Wrapper that safely handles API calls with fallbacks."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.odds_client = None
        self.mlb_client = MLBStatsAPIClient()
        
        if api_key:
            try:
                self.odds_client = TheOddsAPIClient(api_key)
                logger.info("Odds API client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize odds API client: {e}")
        else:
            logger.warning("No API key provided - odds functionality disabled")
    
    def get_todays_odds(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get today's odds with safe error handling."""
        if self.odds_client is None:
            logger.warning("Odds API not available")
            return pd.DataFrame()
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            return self.odds_client.get_home_run_odds_for_date(date)
        except Exception as e:
            logger.error(f"Failed to fetch odds for {date}: {e}")
            return pd.DataFrame()
    
    def get_probable_starters(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get probable starters with safe error handling."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            return self.mlb_client.get_probable_starters(date)
        except Exception as e:
            logger.error(f"Failed to fetch probable starters for {date}: {e}")
            return pd.DataFrame()
    
    def is_odds_available(self) -> bool:
        """Check if odds API is available."""
        return self.odds_client is not None

# Export main classes
__all__ = [
    'TheOddsAPIClient', 'MLBStatsAPIClient', 'SafeAPIClient', 
    'APIError', 'RateLimiter'
]