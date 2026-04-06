import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch

def auto_label(model_path, source_dir, output_dir, conf_thresh):
    """
    Nutzt ein trainiertes YOLO-Modell, um ungelabelte Bilder automatisch zu labeln.
    Erzeugt für jedes Bild eine .txt Datei im YOLO-Format.
    """
    # Automatische Wahl GPU (0), sonst CPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Benutze Geraet: {device}")

    # Das gewählte YOLO-Modell laden
    print(f"Lade Modell von {model_path}...")
    model = YOLO(model_path)

    # Pfade für Bilder und Ausgabe-Labels vorbereiten
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Liste aller Bilder im Quellordner erstellen
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(source_path.glob(ext))
    
    print(f"{len(files)} Bilder in {source_dir} gefunden.")

    count = 0
    # Jedes Bild einzeln verarbeiten
    for img_file in files:
        # Erkennung auf dem Bild ausführen
        results = model(img_file, conf=conf_thresh, verbose=False, device=device)
        
        # Name der Label-Datei generieren (bildname.txt)
        txt_filename = img_file.stem + ".txt"
        txt_path = output_path / txt_filename

        # Ergebnisse in die .txt Datei speichern
        with open(txt_path, "w") as f:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Klasse (ID) und normalisierte Koordinaten (xywhn) abrufen
                    # Format: class_id x_center y_center width height
                    cls = int(box.cls.item())
                    x, y, w, h = box.xywhn[0].tolist()
                    
                    # In Datei speichern (6 Nachkommastellen für Praezision)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        count += 1
        # Fortschrittsanzeige alle 50 Bilder
        if count % 50 == 0:
            print(f"Verarbeitet: {count}/{len(files)} Bilder...")

    print(f"Fertig! Labels wurden gespeichert unter: {output_dir}")

if __name__ == "__main__":
    # Argumente für den Aufruf über die Konsole definieren
    parser = argparse.ArgumentParser(description="Automatisches Labeln von Bildern mit einem YOLO-Modell")
    parser.add_argument("--model", type=str, required=True, help="Pfad zur .pt Modelldatei")
    parser.add_argument("--source", type=str, required=True, help="Ordner mit ungelabelten Bildern")
    parser.add_argument("--output", type=str, required=True, help="Zielordner fuer .txt Labels")
    parser.add_argument("--conf", type=float, default=0.5, help="Konfidenz-Schwellenwert (0.0 bis 1.0)")
    
    args = parser.parse_args()
    
    # Prüfen, ob Modell und Quellordner existieren
    if not os.path.exists(args.model):
        print(f"Fehler: Modell nicht gefunden unter {args.model}")
    elif not os.path.exists(args.source):
        print(f"Fehler: Quellverzeichnis nicht gefunden unter {args.source}")
    else:
        auto_label(args.model, args.source, args.output, args.conf)
