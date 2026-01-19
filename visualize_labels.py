import argparse
import os
import cv2
import glob
from pathlib import Path
import random

def visualize_labels(image_dir, label_dir, output_dir, num_samples=20, classes=None):
    if classes is None:
        # Default classes if not provided
        classes = ["EndBoje", "KursBoje"]
    
    # Setup paths
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(img_path.glob(ext))
    
    if not img_files:
        print(f"No images found in {image_dir}")
        return

    # Select random samples
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    print(f"Visualizing {len(samples)} random samples...")

    # Colors for classes
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] # BGR

    for img_file in samples:
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Find corresponding label file
        txt_file = lbl_path / (img_file.stem + ".txt")
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_idx = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Draw rectangle
                    color = colors[cls_idx % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label_text = classes[cls_idx] if cls_idx < len(classes) else str(cls_idx)
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save output image
        out_file = out_path / ("vis_" + img_file.name)
        cv2.imwrite(str(out_file), img)
        
    print(f"Visualization saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images")
    parser.add_argument("--images", type=str, required=True, help="Folder containing images")
    parser.add_argument("--labels", type=str, required=True, help="Folder containing .txt labels")
    parser.add_argument("--output", type=str, required=True, help="Folder to save visualized images")
    parser.add_argument("--count", type=int, default=20, help="Number of random images to check")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images):
        print(f"Error: Image directory not found at {args.images}")
    elif not os.path.exists(args.labels):
        print(f"Error: Label directory not found at {args.labels}")
    else:
        visualize_labels(args.images, args.labels, args.output, args.count)
