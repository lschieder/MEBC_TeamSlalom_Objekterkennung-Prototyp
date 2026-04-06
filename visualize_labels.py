import argparse
import os
import cv2
from pathlib import Path
import random

def visualize_labels(image_dir, label_dir, output_dir, num_samples=20, classes=None):
    """
    Zeichnet Bounding-Boxes aus YOLO-Label-Dateien (.txt) direkt in die Bilder ein.
    Dient zur schnellen Kontrolle der Labels (Qualitaetsmanagement).
    """
    if classes is None:
        # Klassen
        classes = ["EndBoje", "KursBoje"]
    
    # Pfade vorbereiten
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Alle Bilddateien in einem Ordner suchen
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(img_path.glob(ext))
    
    if not img_files:
        print(f"Keine Bilder gefunden in {image_dir}")
        return

    # Wählt eine zufällige Stichprobe für die Kontrolle aus
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    print(f"Visualisierung von {len(samples)} zufaelligen Bildern...")

    # Farben für die Boundingboxen
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] 

    for img_file in samples:
        # Bild laden
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Den passenden Namen der Label-Datei finden
        txt_file = lbl_path / (img_file.stem + ".txt")
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_idx = int(parts[0])
                    # Das YOLO-Format nutzt normalisierte Koordinaten (0-1)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Umrechnung von normalisierten Koordinaten (YOLO) in Pixel-Koordinaten (für OpenCV)
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Boxen zeichnen
                    color = colors[cls_idx % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Label-Text über der Box platzieren
                    label_text = classes[cls_idx] if cls_idx < len(classes) else str(cls_idx)
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Bild im Ausgabeverzeichnis speichern
        out_file = out_path / ("vis_" + img_file.name)
        cv2.imwrite(str(out_file), img)
        
    print(f"Die Visualisierung wurde gespeichert unter {output_dir}")

if __name__ == "__main__":
    # Steuerung überKommandozeile
    parser = argparse.ArgumentParser(description="Zeigt YOLO Bounding Boxes zur Kontrolle auf Bildern an")
    parser.add_argument("--images", type=str, required=True, help="Ordner mit Bildern")
    parser.add_argument("--labels", type=str, required=True, help="Ordner mit .txt Label-Dateien")
    parser.add_argument("--output", type=str, required=True, help="Ordner fuer Kontroll-Bilder")
    parser.add_argument("--count", type=int, default=20, help="Anzahl der zufaelligen Bilder")
    
    args = parser.parse_args()
    
    # Prüfen, ob die Quellpfade existieren
    if not os.path.exists(args.images):
        print(f"Fehler: Bildverzeichnis nicht gefunden: {args.images}")
    elif not os.path.exists(args.labels):
        print(f"Fehler: Labelverzeichnis nicht gefunden: {args.labels}")
    else:
        visualize_labels(args.images, args.labels, args.output, args.count)
