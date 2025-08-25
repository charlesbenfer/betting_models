"""
Dataset Debugging Script
========================

Comprehensive debugging and analysis script for the baseball home run prediction dataset.
Creates plots, descriptions, and saves all intermediate DataFrames for inspection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetDebugger:
    """Debug and analyze dataset building pipeline."""
    
    def __init__(self, output_dir: str = "debug_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        self.debug_info = {}
        logger.info(f"Debugger initialized. Output directory: {self.output_dir}")
    
    def analyze_dataframe(self, df: pd.DataFrame, name: str, 
                         description: str = "") -> None:
        """Comprehensive analysis of a DataFrame."""
        logger.info(f"Analyzing DataFrame: {name}")
        
        # Always save the DataFrame first
        try:
            self._save_dataframe(df, name)
        except Exception as e:
            logger.error(f"Failed to save DataFrame {name}: {e}")
            return
        
        # Store basic debug info even if analysis fails
        try:
            self.debug_info[name] = {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'dtypes': df.dtypes.value_counts().to_dict(),
                'missing_pct': (df.isnull().sum() / len(df) * 100).mean(),
                'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns)
            }
        except Exception as e:
            logger.warning(f"Failed to store debug info for {name}: {e}")
        
        # Try each analysis step individually
        analysis_steps = [
            ("basic_info", lambda: self._generate_basic_info(df, name, description)),
            ("column_analysis", lambda: self._generate_column_analysis(df, name)),
            ("missing_data", lambda: self._generate_missing_data_analysis(df, name)),
            ("distributions", lambda: self._generate_distribution_plots(df, name)),
            ("correlations", lambda: self._generate_correlation_analysis(df, name)),
            ("time_series", lambda: self._generate_time_series_analysis(df, name))
        ]
        
        for step_name, step_func in analysis_steps:
            try:
                step_func()
                logger.debug(f"Completed {step_name} for {name}")
            except Exception as e:
                logger.warning(f"Failed {step_name} analysis for {name}: {e}")
                continue
        
        # Store debug info
        self.debug_info[name] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).mean(),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns)
        }
    
    def _save_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Save DataFrame in multiple formats."""
        try:
            # Save as parquet (most efficient)
            df.to_parquet(self.output_dir / "data" / f"{name}.parquet", index=False)
            
            # Save as CSV for easy viewing
            df.to_csv(self.output_dir / "data" / f"{name}.csv", index=False)
            
            # Save a sample for quick inspection
            sample_size = min(1000, len(df))
            df.sample(n=sample_size, random_state=42).to_csv(
                self.output_dir / "data" / f"{name}_sample.csv", index=False
            )
            
            logger.info(f"Saved {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Failed to save {name}: {e}")
    
    def _generate_basic_info(self, df: pd.DataFrame, name: str, description: str) -> None:
        """Generate basic information report."""
        report_path = self.output_dir / "reports" / f"{name}_basic_info.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Dataset Analysis Report: {name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if description:
                f.write(f"Description: {description}\n\n")
            
            f.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns\n")
            f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
            
            f.write("Data Types:\n")
            f.write("-" * 20 + "\n")
            for dtype, count in df.dtypes.value_counts().items():
                f.write(f"{dtype}: {count} columns\n")
            
            f.write("\nColumn Names:\n")
            f.write("-" * 20 + "\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:2d}. {col} ({df[col].dtype})\n")
            
            f.write("\nBasic Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(str(df.describe(include='all')))
    
    def _generate_column_analysis(self, df: pd.DataFrame, name: str) -> None:
        """Analyze each column in detail."""
        report_path = self.output_dir / "reports" / f"{name}_column_analysis.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Detailed Column Analysis: {name}\n")
            f.write("=" * 50 + "\n\n")
            
            for col in df.columns:
                f.write(f"Column: {col}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Data Type: {df[col].dtype}\n")
                f.write(f"Non-null Count: {df[col].count():,} / {len(df):,}\n")
                f.write(f"Missing: {df[col].isnull().sum():,} ({df[col].isnull().mean()*100:.1f}%)\n")
                
                if df[col].dtype in ['object', 'category']:
                    unique_count = df[col].nunique()
                    f.write(f"Unique Values: {unique_count:,}\n")
                    if unique_count <= 20:
                        f.write("Value Counts:\n")
                        for val, count in df[col].value_counts().head(10).items():
                            f.write(f"  {val}: {count:,}\n")
                else:
                    f.write(f"Min: {df[col].min()}\n")
                    f.write(f"Max: {df[col].max()}\n")
                    f.write(f"Mean: {df[col].mean():.4f}\n")
                    f.write(f"Median: {df[col].median():.4f}\n")
                    f.write(f"Std: {df[col].std():.4f}\n")
                
                f.write("\n")
    
    def _generate_missing_data_analysis(self, df: pd.DataFrame, name: str) -> None:
        """Analyze missing data patterns."""
        try:
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            
            if missing_data.sum() > 0:
                # Create missing data plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Missing data counts
                missing_data[missing_data > 0].plot(kind='bar', ax=ax1)
                ax1.set_title(f'Missing Data Counts - {name}')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                
                # Missing data percentages
                missing_pct[missing_pct > 0].plot(kind='bar', ax=ax2, color='orange')
                ax2.set_title(f'Missing Data Percentages - {name}')
                ax2.set_ylabel('Percentage')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "plots" / f"{name}_missing_data.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Missing data heatmap for correlations
                if len(df.columns) <= 50:  # Only for manageable number of columns
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, 
                               cmap='viridis', xticklabels=True)
                    plt.title(f'Missing Data Pattern - {name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "plots" / f"{name}_missing_pattern.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Missing data analysis failed for {name}: {e}")
    
    def _generate_distribution_plots(self, df: pd.DataFrame, name: str) -> None:
        """Generate distribution plots for numeric columns."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return
            
            # Limit to most important columns to avoid too many plots
            if len(numeric_cols) > 20:
                # Prioritize columns with interesting names
                priority_patterns = ['hr', 'rate', 'launch', 'speed', 'angle', 'factor']
                priority_cols = []
                other_cols = []
                
                for col in numeric_cols:
                    if any(pattern in col.lower() for pattern in priority_patterns):
                        priority_cols.append(col)
                    else:
                        other_cols.append(col)
                
                numeric_cols = priority_cols[:15] + other_cols[:5]
            
            # Create distribution plots
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    data = df[col].dropna()
                    if len(data) > 0:
                        axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                        axes[i].set_title(f'{col}\n(n={len(data):,})')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                        
                        # Add statistics
                        axes[i].axvline(data.mean(), color='red', linestyle='--', 
                                       label=f'Mean: {data.mean():.3f}')
                        axes[i].axvline(data.median(), color='green', linestyle='--', 
                                       label=f'Median: {data.median():.3f}')
                        axes[i].legend(fontsize=8)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Distribution Analysis - {name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / f"{name}_distributions.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Box plots for key metrics
            key_cols = [col for col in numeric_cols 
                       if any(pattern in col.lower() for pattern in ['hr', 'rate', 'speed', 'angle'])]
            
            if key_cols:
                fig, axes = plt.subplots(1, min(4, len(key_cols)), figsize=(4*min(4, len(key_cols)), 6))
                if len(key_cols) == 1:
                    axes = [axes]
                elif len(key_cols) <= 4:
                    axes = axes if hasattr(axes, '__len__') else [axes]
                
                for i, col in enumerate(key_cols[:4]):
                    data = df[col].dropna()
                    if len(data) > 0:
                        axes[i].boxplot(data)
                        axes[i].set_title(f'{col}\n(n={len(data):,})')
                        axes[i].set_ylabel(col)
                
                plt.suptitle(f'Box Plots - Key Metrics - {name}', fontsize=14)
                plt.tight_layout()
                plt.savefig(self.output_dir / "plots" / f"{name}_boxplots.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Distribution plots failed for {name}: {e}")
    
    def _generate_correlation_analysis(self, df: pd.DataFrame, name: str) -> None:
        """Generate correlation analysis for numeric columns."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols)*0.8)))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=len(numeric_cols) <= 15, 
                       cmap='coolwarm', center=0, square=True, 
                       linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title(f'Correlation Matrix - {name}')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / f"{name}_correlation.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # High correlation pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                high_corr_df.to_csv(
                    self.output_dir / "reports" / f"{name}_high_correlations.csv", 
                    index=False
                )
        except Exception as e:
            logger.warning(f"Correlation analysis failed for {name}: {e}")
    
    def _generate_time_series_analysis(self, df: pd.DataFrame, name: str) -> None:
        """Generate time series analysis if date column exists."""
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        
        if not date_cols:
            return
        
        date_col = date_cols[0]
        
        try:
            df_ts = df.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
            df_ts = df_ts.dropna(subset=[date_col])
            
            if len(df_ts) == 0:
                logger.warning(f"No valid dates found in {date_col} for time series analysis")
                return
            
            df_ts = df_ts.sort_values(date_col)
            
            # Time series overview
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            try:
                # Records over time
                daily_counts = df_ts.groupby(df_ts[date_col].dt.date).size()
                axes[0, 0].plot(daily_counts.index, daily_counts.values)
                axes[0, 0].set_title('Records per Day')
                axes[0, 0].set_xlabel('Date')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].tick_params(axis='x', rotation=45)
            except Exception as e:
                logger.warning(f"Failed to create daily counts plot: {e}")
                axes[0, 0].text(0.5, 0.5, "Daily counts\nplot failed", 
                               transform=axes[0, 0].transAxes, ha='center', va='center')
            
            try:
                # Date range - handle datetime objects more carefully
                start_date = df_ts[date_col].min()
                end_date = df_ts[date_col].max()
                date_range = end_date - start_date
                
                # Convert to string safely
                start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                days_str = str(date_range.days) if hasattr(date_range, 'days') else str(date_range)
                
                axes[0, 1].text(0.1, 0.7, f"Date Range: {days_str} days", 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].text(0.1, 0.5, f"Start: {start_str}", 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].text(0.1, 0.3, f"End: {end_str}", 
                               transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Date Information')
                axes[0, 1].axis('off')
            except Exception as e:
                logger.warning(f"Failed to create date info: {e}")
                axes[0, 1].text(0.5, 0.5, "Date info\nfailed", 
                               transform=axes[0, 1].transAxes, ha='center', va='center')
                axes[0, 1].axis('off')
            
            try:
                # Weekly patterns
                df_ts['day_of_week'] = df_ts[date_col].dt.day_name()
                weekly_counts = df_ts['day_of_week'].value_counts()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_counts = weekly_counts.reindex([day for day in day_order if day in weekly_counts.index])
                
                axes[1, 0].bar(weekly_counts.index, weekly_counts.values)
                axes[1, 0].set_title('Records by Day of Week')
                axes[1, 0].set_xlabel('Day')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
            except Exception as e:
                logger.warning(f"Failed to create weekly patterns: {e}")
                axes[1, 0].text(0.5, 0.5, "Weekly patterns\nplot failed", 
                               transform=axes[1, 0].transAxes, ha='center', va='center')
            
            try:
                # Monthly patterns
                df_ts['month'] = df_ts[date_col].dt.month
                monthly_counts = df_ts['month'].value_counts().sort_index()
                axes[1, 1].bar(monthly_counts.index, monthly_counts.values)
                axes[1, 1].set_title('Records by Month')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].set_ylabel('Count')
            except Exception as e:
                logger.warning(f"Failed to create monthly patterns: {e}")
                axes[1, 1].text(0.5, 0.5, "Monthly patterns\nplot failed", 
                               transform=axes[1, 1].transAxes, ha='center', va='center')
            
            plt.suptitle(f'Time Series Analysis - {name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / f"{name}_time_series.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Time series analysis failed for {name}: {e}")
    
    def debug_dataset_pipeline(self, builder, force_rebuild: bool = False):
        """Debug the entire dataset building pipeline step by step."""
        logger.info("Starting comprehensive dataset pipeline debugging...")
        
        results = {}
        
        try:
            # Step 1: Raw Statcast Data
            logger.info("Fetching raw Statcast data...")
            try:
                statcast_data = builder.statcast_processor.fetch_statcast_data(
                    builder.start_date, builder.end_date, use_cache=not force_rebuild
                )
                self.analyze_dataframe(statcast_data, "01_raw_statcast", 
                                     "Raw Statcast data from Baseball Savant")
                results['statcast_data'] = statcast_data
            except Exception as e:
                logger.error(f"Failed to fetch Statcast data: {e}")
                return None
            
            # Step 2: Batter-Game Features
            logger.info("Creating batter-game features...")
            try:
                batter_games = builder._create_batter_game_features(statcast_data)
                self.analyze_dataframe(batter_games, "02_batter_games", 
                                     "Batter-game aggregations with rolling features")
                results['batter_games'] = batter_games
            except Exception as e:
                logger.error(f"Failed to create batter-game features: {e}")
                # Save what we have so far
                self._generate_pipeline_summary()
                raise
            
            # Step 3: Handedness Splits
            logger.info("Adding handedness splits...")
            try:
                batter_games_hand = builder.handedness_calculator.calculate_handedness_splits(
                    statcast_data, batter_games
                )
                self.analyze_dataframe(batter_games_hand, "03_with_handedness", 
                                     "Batter-games with handedness split features")
                results['batter_games_hand'] = batter_games_hand
            except Exception as e:
                logger.error(f"Failed to add handedness splits: {e}")
                # Continue with previous step's data
                batter_games_hand = batter_games
                results['batter_games_hand'] = batter_games_hand
            
            # Step 4: Pitcher Features
            logger.info("Adding pitcher features...")
            try:
                batter_games_pitcher = builder._add_pitcher_features(statcast_data, batter_games_hand)
                self.analyze_dataframe(batter_games_pitcher, "04_with_pitcher", 
                                     "Batter-games with pitcher quality features")
                results['batter_games_pitcher'] = batter_games_pitcher
            except Exception as e:
                logger.error(f"Failed to add pitcher features: {e}")
                # Continue with previous step's data
                batter_games_pitcher = batter_games_hand
                results['batter_games_pitcher'] = batter_games_pitcher
            
            # Step 5: Pitch Matchup Features
            logger.info("Adding pitch matchup features...")
            try:
                batter_games_matchup = builder.matchup_calculator.calculate_pitch_matchup_features(
                    statcast_data, batter_games_pitcher
                )
                self.analyze_dataframe(batter_games_matchup, "05_with_matchup", 
                                     "Batter-games with pitch matchup features")
                results['batter_games_matchup'] = batter_games_matchup
            except Exception as e:
                logger.error(f"Failed to add pitch matchup features: {e}")
                # Continue with previous step's data
                batter_games_matchup = batter_games_pitcher
                results['batter_games_matchup'] = batter_games_matchup
            
            # Step 6: Weather Features
            logger.info("Adding weather features...")
            try:
                batter_games_weather = builder._add_weather_features(batter_games_matchup)
                self.analyze_dataframe(batter_games_weather, "06_with_weather", 
                                     "Batter-games with weather features")
                results['batter_games_weather'] = batter_games_weather
            except Exception as e:
                logger.error(f"Failed to add weather features: {e}")
                # Continue with previous step's data
                batter_games_weather = batter_games_matchup
                results['batter_games_weather'] = batter_games_weather
            
            # Step 7: Final Dataset
            logger.info("Finalizing dataset...")
            try:
                final_dataset = builder._finalize_dataset(batter_games_weather)
                self.analyze_dataframe(final_dataset, "07_final_dataset", 
                                     "Final processed dataset ready for modeling")
                results['final_dataset'] = final_dataset
            except Exception as e:
                logger.error(f"Failed to finalize dataset: {e}")
                # Use the last successful step as final
                final_dataset = batter_games_weather
                results['final_dataset'] = final_dataset
            
            # Generate pipeline summary
            self._generate_pipeline_summary()
            
            logger.info("Dataset pipeline debugging completed!")
            return results.get('final_dataset', None)
            
        except Exception as e:
            logger.error(f"Pipeline debugging failed: {e}")
            self._generate_error_report(str(e))
            # Return what we have
            return results.get('final_dataset', results.get('batter_games_weather', 
                              results.get('batter_games_matchup', results.get('batter_games_pitcher', 
                              results.get('batter_games_hand', results.get('batter_games', None))))))
    
    def _generate_pipeline_summary(self):
        """Generate overall pipeline summary."""
        summary_path = self.output_dir / "reports" / "pipeline_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Dataset Pipeline Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Pipeline Stages:\n")
            f.write("-" * 20 + "\n")
            
            for stage, info in self.debug_info.items():
                f.write(f"\n{stage}:\n")
                f.write(f"  Shape: {info['shape'][0]:,} rows × {info['shape'][1]:,} columns\n")
                f.write(f"  Memory: {info['memory_usage']:.2f} MB\n")
                f.write(f"  Missing Data: {info['missing_pct']:.1f}% average\n")
                f.write(f"  Numeric Columns: {info['numeric_cols']}\n")
                f.write(f"  Categorical Columns: {info['categorical_cols']}\n")
            
            f.write(f"\nTotal Files Generated:\n")
            f.write(f"  Data files: {len(list((self.output_dir / 'data').glob('*')))}\n")
            f.write(f"  Plot files: {len(list((self.output_dir / 'plots').glob('*')))}\n")
            f.write(f"  Report files: {len(list((self.output_dir / 'reports').glob('*')))}\n")
    
    def _generate_error_report(self, error_msg: str):
        """Generate error report."""
        error_path = self.output_dir / "reports" / "error_report.txt"
        
        with open(error_path, 'w') as f:
            f.write("Pipeline Error Report\n")
            f.write("=" * 25 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Error: {error_msg}\n\n")
            f.write("Completed Stages:\n")
            f.write("-" * 20 + "\n")
            
            for stage, info in self.debug_info.items():
                f.write(f"{stage}: {info['shape']}\n")


def main():
    """Main debugging function."""
    # You'll need to import your actual modules here
    try:
        from dataset_builder import PregameDatasetBuilder
        from config import config
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure your dataset_builder and config modules are available")
        return
    
    # Initialize debugger
    debugger = DatasetDebugger("debug_output")
    
    # Initialize dataset builder
    # Adjust these parameters as needed
    builder = PregameDatasetBuilder(
        start_date="2024-04-01",  # Adjust as needed
        end_date="2024-04-30",    # Adjust as needed
        weather_csv_path=None     # Add path if you have weather data
    )
    
    # Run debugging
    try:
        final_dataset = debugger.debug_dataset_pipeline(builder, force_rebuild=True)
        
        if final_dataset is not None:
            print(f"\nDebugging completed successfully!")
            print(f"Output directory: {debugger.output_dir}")
            print(f"Final dataset shape: {final_dataset.shape}")
        else:
            print(f"\nDebugging completed with issues!")
            print(f"Output directory: {debugger.output_dir}")
            print(f"Check the debug files for partial results")
        
        # Print summary of what was generated
        data_files = list((debugger.output_dir / "data").glob("*.csv"))
        plot_files = list((debugger.output_dir / "plots").glob("*.png"))
        report_files = list((debugger.output_dir / "reports").glob("*.txt"))
        
        print(f"\nGenerated files:")
        print(f"  Data files: {len(data_files)}")
        print(f"  Plot files: {len(plot_files)}")
        print(f"  Report files: {len(report_files)}")
        
    except Exception as e:
        print(f"Debugging failed: {e}")
        print(f"Check the debug output directory for partial results: {debugger.output_dir}")
        
        # Still print what we managed to generate
        try:
            data_files = list((debugger.output_dir / "data").glob("*.csv"))
            plot_files = list((debugger.output_dir / "plots").glob("*.png"))
            report_files = list((debugger.output_dir / "reports").glob("*.txt"))
            
            print(f"\nPartial results generated:")
            print(f"  Data files: {len(data_files)}")
            print(f"  Plot files: {len(plot_files)}")
            print(f"  Report files: {len(report_files)}")
        except:
            pass


if __name__ == "__main__":
    main()