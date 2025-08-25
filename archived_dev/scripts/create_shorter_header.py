"""
Create Shorter Header for Blog
============================

Creates a horizontally optimized header that fits better in blog layout.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_shorter_header():
    """Create a shorter, wider header that shows all content."""
    
    # Much shorter and narrower figure - optimized for web header
    fig, ax = plt.subplots(figsize=(8, 3))  # Reduced width to 8 for compact blog header
    
    # Your actual data
    categories = ['Situational', 'Matchup', 'Interactions', 'Streaks', 
                  'Core', 'Recent Form', 'Ballpark', 'Temporal', 'Weather']
    correlations = [0.174, 0.040, 0.038, 0.037, 0.033, 0.025, 0.023, 0.023, 0.014]
    
    # Create gradient colors manually
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    # Horizontal layout to maximize space usage
    bars = ax.barh(categories, correlations, color=colors, 
                   edgecolor='white', linewidth=1.5, height=0.7)
    
    # Add values on bars - smaller font for compact layout
    for bar, corr in zip(bars, correlations):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Compact title and labels
    ax.set_xlabel('Correlation with Home Run Probability', fontsize=11, fontweight='bold')
    ax.set_title('255+ MLB Features: Predictive Power by Category | 92.7% Implementation Success', 
                 fontsize=13, fontweight='bold', pad=10)  # Reduced padding
    
    # Tighter grid and layout
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    
    # Tighter margins
    plt.tight_layout(pad=0.5)
    
    # Save directly to GitHub Pages assets folder
    output_path = '/home/charlesbenfer/charlesbenfer.github.io/assets/img/feature_engineering_header_final.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Shorter header saved to: {output_path}")
    plt.close()

def main():
    """Generate shorter header."""
    print("🖼️  Creating shorter header for blog...")
    create_shorter_header()
    print("🎉 Final header created and saved to GitHub Pages assets!")

if __name__ == "__main__":
    main()