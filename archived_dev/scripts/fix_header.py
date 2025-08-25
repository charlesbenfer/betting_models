"""
Fix Header Visualization Issues
=============================

Corrects arrow positioning and text wrapping issues.
"""

import matplotlib.pyplot as plt

def create_fixed_header():
    """Create improved pipeline header with corrected arrows and text."""
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Pipeline stages with improved text
    stages = ['Raw Data', 'Feature\nEngineering', '255+ Features', 'ML Models', 'Predictions']
    positions = [1, 3.5, 6, 8.5, 11]
    colors = ['#3498DB', '#9B59B6', '#E74C3C', '#F39C12', '#27AE60']
    
    # Draw boxes
    for pos, stage, color in zip(positions, stages, colors):
        rect = plt.Rectangle((pos-0.6, 1), 1.2, 0.8, 
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos, 1.4, stage, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
    
    # Draw arrows with proper positioning (tips end at start of next box)
    for i in range(len(positions)-1):
        start_x = positions[i] + 0.6  # Right edge of current box
        end_x = positions[i+1] - 0.6  # Left edge of next box
        arrow_length = end_x - start_x - 0.2  # Leave space for arrow head
        
        ax.arrow(start_x, 1.4, arrow_length, 0,
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
    plt.savefig('/home/charlesbenfer/betting_models/header_fixed.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Fixed header saved as header_fixed.png")
    plt.close()

if __name__ == "__main__":
    create_fixed_header()
    print("🎨 Header fixes applied!")
    print("  - Arrows now end at rectangle edges")
    print("  - 'Feature Engineering' split into two lines")