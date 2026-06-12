import cv2
import numpy as np

def draw_grid_overlay(image, rows=3, cols=3):
    """Draws faint grid lines on the image."""
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]
    
    # Draw vertical lines
    for i in range(1, cols):
        x = int(w * (i / cols))
        cv2.line(img_with_grid, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        
    # Draw horizontal lines
    for i in range(1, rows):
        y = int(h * (i / rows))
        cv2.line(img_with_grid, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
        
    # Make lines faint using weighted add
    alpha = 0.3
    cv2.addWeighted(img_with_grid, alpha, image, 1 - alpha, 0, image)
    return image

def draw_empty_zones(image, empty_zones, rows=3, cols=3):
    """Highlights empty zones in a red overlay."""
    img_with_zones = image.copy()
    h, w = img_with_zones.shape[:2]
    
    cell_w = w / cols
    cell_h = h / rows
    
    for (r, c) in empty_zones:
        x1 = int(c * cell_w)
        y1 = int(r * cell_h)
        x2 = int((c + 1) * cell_w)
        y2 = int((r + 1) * cell_h)
        
        # Red overlay for empty zone (RGB format)
        cv2.rectangle(img_with_zones, (x1, y1), (x2, y2), (255, 0, 0), -1)
        
    alpha = 0.4
    cv2.addWeighted(img_with_zones, alpha, image, 1 - alpha, 0, image)
    return image

def format_bbox_label(class_name, confidence):
    """Returns formatted string for bounding box label."""
    return f"{class_name} {confidence:.2f}"

def resize_for_display(image, max_width=800):
    """Resizes image keeping aspect ratio if wider than max_width."""
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        image = cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return image
