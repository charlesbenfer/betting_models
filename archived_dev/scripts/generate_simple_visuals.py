"""
Generate Simple Professional Visualizations for Blog Post
========================================================

Creates clean, professional visualizations using your actual data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set clean style
plt.style.use('default')

def create_thumbnail():
    """Create feature correlation bar chart."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Your actual data
    categories = ['Situational', 'Matchup', 'Interactions', 'Streaks', 
                  'Core', 'Recent Form', 'Ballpark', 'Temporal', 'Weather']
    correlations = [0.174, 0.040, 0.038, 0.037, 0.033, 0.025, 0.023, 0.023, 0.014]
    
    # Create gradient colors manually
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    bars = ax.barh(categories, correlations, color=colors, 
                   edgecolor='white', linewidth=2)
    
    # Add values on bars
    for bar, corr in zip(bars, correlations):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Average Correlation with Home Run Probability', fontsize=12, fontweight='bold')
    ax.set_title('MLB Feature Engineering: 255+ Features by Category\n92.7% Implementation Success', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/thumbnail.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Thumbnail saved")
    plt.close()

def create_header():
    """Create pipeline header diagram."""
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Pipeline stages
    stages = ['Raw Data', 'Feature Engineering', '255+ Features', 'ML Models', 'Predictions']
    positions = [1, 3.5, 6, 8.5, 11]
    colors = ['#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#27AE60']
    
    # Draw boxes
    for pos, stage, color in zip(positions, stages, colors):
        rect = plt.Rectangle((pos-0.6, 1), 1.2, 0.8, 
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos, 1.4, stage, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
    
    # Draw arrows
    for i in range(len(positions)-1):
        ax.arrow(positions[i]+0.6, 1.4, positions[i+1]-positions[i]-1.2, 0,
                head_width=0.15, head_length=0.2, fc='black', ec='black')
    
    # Add metrics
    metrics = ['4 Years Data', '8 Categories', '147,169× Faster', '0.75-0.80 AUC']
    for i, metric in enumerate(metrics):
        ax.text(2 + i*2.5, 0.4, metric, ha='center', fontsize=10, 
               fontweight='bold', color='#2C3E50')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    ax.set_title('Advanced MLB Home Run Prediction Pipeline', 
                fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/header.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Header saved")
    plt.close()

def create_performance():
    """Create performance comparison."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model comparison
    models = ['Baseline', 'With 255+ Features']
    scores = [0.73, 0.78]
    colors = ['#95A5A6', '#27AE60']
    
    bars = ax1.bar(models, scores, color=colors, edgecolor='white', linewidth=2)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylim(0.5, 0.85)
    ax1.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax1.set_title('Model Performance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Feature impact
    categories = ['Situational', 'Matchup', 'Interactions', 'Streaks', 'Core']
    impacts = [0.174, 0.040, 0.038, 0.037, 0.033]
    
    ax2.barh(categories, impacts, color='#3498DB', edgecolor='white', linewidth=2)
    ax2.set_xlabel('Correlation', fontweight='bold')
    ax2.set_title('Top Feature Categories', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('255+ Feature Engineering Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/performance.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Performance chart saved")
    plt.close()

def main():
    """Generate all visualizations."""
    print("🎨 Generating blog visualizations...")
    
    create_thumbnail()
    create_header() 
    create_performance()
    
    print("\n🎉 All visualizations generated!")
    print("\nFiles created:")
    print("  📊 thumbnail.png - Feature correlation chart")
    print("  🎨 header.png - Pipeline diagram")
    print("  📈 performance.png - Performance comparison")

if __name__ == "__main__":
    main()