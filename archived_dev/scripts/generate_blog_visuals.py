"""
Generate Professional Visualizations for MLB Feature Engineering Blog Post
=========================================================================

Creates thumbnail and header images showcasing the 255+ feature pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib import colors

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_feature_correlation_thumbnail():
    """Create a stunning bar chart showing feature category performance."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Actual data from your testing
    categories = ['Situational\nContext', 'Matchup\nAnalysis', 'Feature\nInteractions', 
                  'Streak/\nMomentum', 'Core\nStats', 'Recent\nForm', 
                  'Ballpark\nFactors', 'Temporal/\nFatigue', 'Weather\nImpact']
    correlations = [0.174, 0.040, 0.038, 0.037, 0.033, 0.025, 0.023, 0.023, 0.014]
    feature_counts = [33, 17, 35, 29, 17, 24, 35, 41, 20]
    
    # Create gradient colors
    cmap = plt.cm.get_cmap('viridis')
    colors_list = [cmap(i/len(categories)) for i in range(len(categories))]
    
    # Create bars
    bars = ax.barh(categories, correlations, color=colors_list, edgecolor='white', linewidth=2)
    
    # Add feature counts as text
    for i, (bar, count) in enumerate(zip(bars, feature_counts)):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{count} features', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Average Correlation with Home Run Probability', fontsize=14, fontweight='bold')
    ax.set_title('MLB Feature Engineering: 255+ Features Across 9 Categories\nPredictive Power Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        width = bar.get_width()
        ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Grid and aesthetics
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 0.20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add performance badge
    badge = FancyBboxPatch((0.15, 8.2), 0.045, 0.6, 
                          boxstyle="round,pad=0.02", 
                          facecolor='#2ecc71', edgecolor='white', linewidth=2)
    ax.add_patch(badge)
    ax.text(0.1725, 8.5, '92.7%', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    ax.text(0.1725, 8.85, 'Implementation', fontsize=8, 
            ha='center', va='center', color='#333')
    
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/blog_thumbnail.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Thumbnail saved as blog_thumbnail.png")
    plt.show()

def create_pipeline_header():
    """Create a wide header image showing the feature engineering pipeline."""
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Create gradient background
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    gradient = np.hstack((gradient, gradient))
    
    ax.imshow(gradient.T, extent=[0, 10, 0, 3], aspect='auto', 
              cmap='coolwarm', alpha=0.3)
    
    # Pipeline stages
    stages = [
        {'name': 'Raw\nStatcast\nData', 'pos': 0.5, 'color': '#3498db'},
        {'name': 'Feature\nEngineering\nPipeline', 'pos': 2.5, 'color': '#9b59b6'},
        {'name': '255+\nEngineered\nFeatures', 'pos': 5, 'color': '#e74c3c'},
        {'name': 'ML Models\n(XGBoost,\nRF, LR)', 'pos': 7.5, 'color': '#f39c12'},
        {'name': 'Production\nPredictions', 'pos': 9.5, 'color': '#2ecc71'}
    ]
    
    # Draw pipeline boxes
    for stage in stages:
        fancy_box = FancyBboxPatch((stage['pos']-0.4, 1.2), 0.8, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=stage['color'], 
                                  edgecolor='white',
                                  linewidth=3,
                                  alpha=0.9)
        ax.add_patch(fancy_box)
        ax.text(stage['pos'], 1.5, stage['name'], 
               fontsize=12, fontweight='bold', 
               ha='center', va='center', color='white')
    
    # Draw arrows between stages
    arrow_y = 1.5
    for i in range(len(stages)-1):
        ax.arrow(stages[i]['pos']+0.4, arrow_y, 
                stages[i+1]['pos']-stages[i]['pos']-0.8, 0,
                head_width=0.1, head_length=0.1, 
                fc='#34495e', ec='#34495e', linewidth=2)
    
    # Add feature categories in the middle
    feature_categories = [
        'Matchup Analysis', 'Situational Context', 'Weather Impact',
        'Recent Form', 'Streak/Momentum', 'Ballpark Factors',
        'Temporal/Fatigue', 'Feature Interactions'
    ]
    
    for i, cat in enumerate(feature_categories):
        y_pos = 0.3 + (i % 2) * 0.3
        x_pos = 1.5 + (i // 2) * 0.5
        ax.text(x_pos, y_pos, cat, fontsize=8, 
               ha='center', va='center', 
               style='italic', color='#2c3e50', alpha=0.8)
    
    # Add performance metrics
    metrics_text = [
        '147,169× Speed Improvement',
        '4 Years MLB Data',
        '0.75-0.80 ROC-AUC',
        'Sub-millisecond Inference'
    ]
    
    for i, metric in enumerate(metrics_text):
        ax.text(2.5 + i*2, 2.5, metric, fontsize=10, 
               fontweight='bold', ha='center', 
               color='#2c3e50', alpha=0.9)
    
    # Title
    ax.text(5, 0.2, 'Advanced MLB Home Run Prediction Pipeline', 
           fontsize=24, fontweight='bold', ha='center', color='#2c3e50')
    
    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/blog_header.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Header saved as blog_header.png")
    plt.show()

def create_performance_comparison():
    """Create a comparison chart showing model improvements."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC-AUC Comparison
    models = ['Baseline\nModel', 'With 255+\nFeatures']
    roc_scores = [0.73, 0.78]
    colors_bars = ['#95a5a6', '#2ecc71']
    
    bars1 = ax1.bar(models, roc_scores, color=colors_bars, 
                    edgecolor='white', linewidth=3)
    
    # Add value labels
    for bar, score in zip(bars1, roc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax1.set_ylim(0.5, 0.85)
    ax1.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Feature Impact Breakdown
    categories = ['Core', 'Matchup', 'Situational', 'Weather', 'Form', 
                 'Streaks', 'Ballpark', 'Temporal', 'Interactions']
    impacts = [0.033, 0.040, 0.174, 0.014, 0.025, 0.037, 0.023, 0.023, 0.038]
    
    # Sort by impact
    sorted_data = sorted(zip(categories, impacts), key=lambda x: x[1], reverse=True)
    categories_sorted = [x[0] for x in sorted_data]
    impacts_sorted = [x[1] for x in sorted_data]
    
    # Create gradient colors
    cmap = plt.cm.get_cmap('plasma')
    colors_gradient = [cmap(i/len(categories)) for i in range(len(categories))]
    
    bars2 = ax2.barh(categories_sorted, impacts_sorted, color=colors_gradient,
                     edgecolor='white', linewidth=2)
    
    ax2.set_xlabel('Average Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Category Impact', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add impact improvement arrow
    improvement = ((0.78 - 0.73) / 0.73) * 100
    ax1.annotate(f'+{improvement:.1f}%', 
                xy=(1, 0.78), xytext=(0.5, 0.76),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
                fontsize=16, fontweight='bold', color='#e74c3c')
    
    plt.suptitle('255+ Feature Engineering Impact Analysis', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/blog_performance.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Performance comparison saved as blog_performance.png")
    plt.show()

def create_complexity_visualization():
    """Create a visualization showing the optimization improvements."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data for complexity improvements
    operations = ['Pitcher\nSimilarity', 'Recent Form\nCalculation', 'Matchup\nDatabase', 
                 'Feature\nPipeline', 'Weather API\nFallback']
    before_times = [1.5, 2.0, 0.4, 6.0, 0.5]  # hours or seconds
    after_times = [0.005, 0.25, 0.000003, 3.5, 0.001]  # optimized times
    improvements = [300, 8, 147169, 1.7, 500]  # x improvement
    
    x = np.arange(len(operations))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, before_times, width, label='Before Optimization',
                   color='#e74c3c', alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, after_times, width, label='After Optimization',
                   color='#2ecc71', alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add improvement labels
    for i, (op, improve) in enumerate(zip(operations, improvements)):
        y_pos = max(before_times[i], after_times[i]) + 0.2
        if improve > 100:
            label = f'{improve:,}×\nfaster'
        else:
            label = f'{improve:.1f}×\nfaster'
        ax.text(i, y_pos, label, ha='center', va='bottom',
               fontsize=11, fontweight='bold', color='#2c3e50')
    
    # Styling
    ax.set_ylabel('Execution Time (relative scale)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Engineering Optimization Results\nFrom O(n²) to O(n) Complexity', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total improvement badge
    total_badge = FancyBboxPatch((3.5, 5.5), 1.4, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#3498db', 
                                edgecolor='white',
                                linewidth=3, alpha=0.9)
    ax.add_patch(total_badge)
    ax.text(4.2, 5.9, '50-70% Total\nSpeed Improvement', 
           fontsize=12, fontweight='bold',
           ha='center', va='center', color='white')
    
    ax.set_ylim(0, 7)
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/blog_optimization.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Optimization visualization saved as blog_optimization.png")
    plt.show()

def main():
    """Generate all visualizations for the blog post."""
    print("🎨 Generating blog visualizations...")
    print("-" * 50)
    
    # Create all visualizations
    create_feature_correlation_thumbnail()
    create_pipeline_header()
    create_performance_comparison()
    create_complexity_visualization()
    
    print("-" * 50)
    print("🎉 All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  📊 blog_thumbnail.png - Feature correlation bar chart")
    print("  🎨 blog_header.png - Pipeline flow diagram")
    print("  📈 blog_performance.png - Performance comparison charts")
    print("  ⚡ blog_optimization.png - Optimization improvements")
    print("\nYou can find them in: /home/charlesbenfer/betting_models/")

if __name__ == "__main__":
    main()