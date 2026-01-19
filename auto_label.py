import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch

def auto_label(model_path, source_dir, output_dir, conf_thresh):
    # Check device
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Setup paths
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(source_path.glob(ext))
    
    print(f"Found {len(files)} images in {source_dir}")

    count = 0
    for img_file in files:
        # Run inference
        results = model(img_file, conf=conf_thresh, verbose=False, device=device)
        
        # Prepare label file path
        txt_filename = img_file.stem + ".txt"
        txt_path = output_path / txt_filename

        # Write labels
        with open(txt_path, "w") as f:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class and normalized coordinates (xywhn) directly from YOLO
                    cls = int(box.cls.item())
                    x, y, w, h = box.xywhn[0].tolist()
                    
                    # Write to file: class x_center y_center width height
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        count += 1
        if count % 50 == 0:
            print(f"Processed {count}/{len(files)} images...")

    print(f"Done! Labels saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using a YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--source", type=str, required=True, help="Folder containing images to label")
    parser.add_argument("--output", type=str, required=True, help="Folder to save .txt labels")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
    elif not os.path.exists(args.source):
        print(f"Error: Source directory not found at {args.source}")
    else:
        auto_label(args.model, args.source, args.output, args.conf)
