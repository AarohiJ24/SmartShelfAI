import pandas as pd
import numpy as np

def product_count(detections):
    """Count total detected items."""
    return len(detections)

def shelf_occupancy(detections, image_width, image_height):
    """
    Calculate % of shelf area covered by bounding boxes.
    Calculates union of all bounding boxes to avoid double counting overlapping areas.
    """
    if not detections:
        return 0.0

    # Create a blank mask for the image
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        # Ensure coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        mask[y1:y2, x1:x2] = 1

    covered_area = np.sum(mask)
    total_area = image_width * image_height
    
    if total_area == 0:
        return 0.0
        
    occupancy_pct = (covered_area / total_area) * 100.0
    return round(occupancy_pct, 2)

def stock_status(occupancy_pct, low_stock_threshold=30):
    """
    Return stock status based on occupancy percentage.
    >=50% -> "Well Stocked"
    low_stock_threshold-50% -> "Low Stock"
    <low_stock_threshold% -> "Critical - Restock Needed"
    """
    if occupancy_pct >= 50:
        return "Well Stocked"
    elif occupancy_pct >= low_stock_threshold:
        return "Low Stock"
    else:
        return "Critical - Restock Needed"

def empty_zone_detection(detections, image_width, image_height, rows=3, cols=3):
    """
    Divide shelf into a grid (default 3x3).
    Identify grid cells with zero detections.
    Return list of empty zone coordinates like (row, col).
    """
    cell_w = image_width / cols
    cell_h = image_height / rows
    
    # Initialize grid with zeros
    grid = np.zeros((rows, cols), dtype=int)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # Find the center of the bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Determine which cell the center falls into
        c = int(cx / cell_w)
        r = int(cy / cell_h)
        
        # Ensure it is within bounds
        r = min(max(r, 0), rows - 1)
        c = min(max(c, 0), cols - 1)
        
        grid[r, c] += 1
        
    empty_zones = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:
                empty_zones.append((r, c))
                
    return empty_zones

def get_zone_name(bbox, image_width, image_height, rows=3, cols=3):
    """Helper to get zone name for a specific bounding box."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    cell_w = image_width / cols
    cell_h = image_height / rows
    
    c = int(cx / cell_w)
    r = int(cy / cell_h)
    
    r = min(max(r, 0), rows - 1)
    c = min(max(c, 0), cols - 1)
    
    return f"Zone ({r},{c})"

def generate_report(detections, image_width, image_height):
    """
    Return a pandas DataFrame summary with columns:
    [Product Class, Confidence, BBox Location, Zone]
    """
    if not detections:
        return pd.DataFrame(columns=["Product Class", "Confidence", "BBox Location", "Zone"])
        
    data = []
    for det in detections:
        bbox_str = f"[{int(det['bbox'][0])}, {int(det['bbox'][1])}, {int(det['bbox'][2])}, {int(det['bbox'][3])}]"
        zone = get_zone_name(det['bbox'], image_width, image_height)
        
        data.append({
            "Product Class": det['class'],
            "Confidence": f"{det['confidence']:.2f}",
            "BBox Location": bbox_str,
            "Zone": zone
        })
        
    return pd.DataFrame(data)
