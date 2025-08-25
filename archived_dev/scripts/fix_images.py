"""
Fix Blog Images - Resize Header and Create Square Thumbnail
========================================================

Creates properly sized images for blog post.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def resize_header_image():
    """Resize the original thumbnail to 75% for header use."""
    try:
        # Load the original thumbnail
        img = Image.open('/home/charlesbenfer/betting_models/thumbnail.png')
        
        # Get current size
        width, height = img.size
        print(f"Original size: {width}x{height}")
        
        # Resize to 75%
        new_width = int(width * 0.75)
        new_height = int(height * 0.75)
        
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save resized version
        resized_img.save('/home/charlesbenfer/betting_models/header_resized.png', 'PNG', quality=95)
        print(f"✅ Header resized to: {new_width}x{new_height}")
        print("   Saved as: header_resized.png")
        
    except Exception as e:
        print(f"❌ Error resizing header: {e}")

def create_square_thumbnail():
    """Create a new square thumbnail image."""
    
    fig, ax = plt.subplots(figsize=(8, 8))  # Square aspect ratio
    
    # Key metrics in a circle layout
    center_x, center_y = 0.5, 0.5
    
    # Main title in center
    ax.text(center_x, center_y + 0.15, '255+', 
           fontsize=48, fontweight='bold', ha='center', va='center',
           color='#2C3E50')
    ax.text(center_x, center_y + 0.05, 'Features', 
           fontsize=24, fontweight='bold', ha='center', va='center',
           color='#2C3E50')
    
    # Surrounding metrics
    metrics = [
        {'text': '92.7%\nImplemented', 'angle': 0, 'color': '#27AE60'},
        {'text': '8\nCategories', 'angle': 45, 'color': '#3498DB'},
        {'text': '0.75-0.80\nROC-AUC', 'angle': 90, 'color': '#E74C3C'},
        {'text': '147,169×\nFaster', 'angle': 135, 'color': '#F39C12'},
        {'text': '4 Years\nData', 'angle': 180, 'color': '#9B59B6'},
        {'text': 'Production\nReady', 'angle': 225, 'color': '#1ABC9C'},
        {'text': 'Real-time\nInference', 'angle': 270, 'color': '#E67E22'},
        {'text': 'Sports\nBetting', 'angle': 315, 'color': '#34495E'}
    ]
    
    radius = 0.3
    for metric in metrics:
        angle_rad = np.radians(metric['angle'])
        x = center_x + radius * np.cos(angle_rad)
        y = center_y + radius * np.sin(angle_rad)
        
        ax.text(x, y, metric['text'], 
               fontsize=10, fontweight='bold', ha='center', va='center',
               color=metric['color'],
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=metric['color'], linewidth=2, alpha=0.9))
    
    # Add subtle background circle
    circle = plt.Circle((center_x, center_y), 0.45, 
                       fill=False, edgecolor='#BDC3C7', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    
    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Subtle title
    ax.text(center_x, 0.05, 'MLB Feature Engineering', 
           fontsize=14, fontweight='bold', ha='center', va='center',
           color='#7F8C8D', style='italic')
    
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/thumbnail_square.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Square thumbnail created: thumbnail_square.png")
    plt.close()

def create_simple_square_thumbnail():
    """Create a simpler square thumbnail."""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Background gradient
    gradient = np.linspace(0, 1, 100)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', 
              cmap='viridis', alpha=0.2)
    
    # Main content
    ax.text(0.5, 0.7, '255+', fontsize=42, fontweight='bold', 
           ha='center', va='center', color='#2C3E50')
    ax.text(0.5, 0.55, 'Features', fontsize=18, fontweight='bold',
           ha='center', va='center', color='#2C3E50')
    
    # Key stats
    ax.text(0.5, 0.4, '92.7% Implementation Success', fontsize=11, 
           ha='center', va='center', color='#27AE60', fontweight='bold')
    ax.text(0.5, 0.32, '8 Feature Categories', fontsize=11,
           ha='center', va='center', color='#3498DB', fontweight='bold') 
    ax.text(0.5, 0.24, '147,169× Speed Improvement', fontsize=11,
           ha='center', va='center', color='#E74C3C', fontweight='bold')
    
    # Footer
    ax.text(0.5, 0.1, 'MLB Home Run Prediction', fontsize=12,
           ha='center', va='center', color='#7F8C8D', style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/charlesbenfer/betting_models/thumbnail_simple_square.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Simple square thumbnail created: thumbnail_simple_square.png")
    plt.close()

def main():
    """Generate fixed images."""
    print("🖼️  Fixing blog images...")
    print("-" * 40)
    
    # Resize header
    resize_header_image()
    
    # Create square thumbnails
    create_square_thumbnail()
    create_simple_square_thumbnail()
    
    print("-" * 40)
    print("🎉 Image fixes complete!")
    print("\nGenerated files:")
    print("  📏 header_resized.png - 75% size for header")
    print("  🟦 thumbnail_square.png - Circular layout design")
    print("  🟦 thumbnail_simple_square.png - Clean simple design")

if __name__ == "__main__":
    main()