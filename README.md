# MEBC Objekterkennung - Kurzanleitung

Dieses Repository enthält Tools und Skripte zur Objekterkennung von Bojen für das **MEBC TeamSlalom Projekt**. 
Es ermöglicht das automatisierte Labeln von Daten, das Training von YOLOv11-Modellen sowie die Live-Erkennung über Videoquellen oder OBS Virtual Camera.

### 1. Daten vorbereiten
Bilder in `images/`, Labels in `labels/` ablegen.
```bash
python train_val_split.py --datapath="C:\Objectdetection\Objectdetection" --train_pct=.8
```

### 2. Training starten
```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

### 3. Erkennung ausführen (Webcam/OBS)
```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source usb0
```

### 4. Weitere Tools
* **Auto-Label:** `python auto_label.py --model [Pfad] --source [Bilder] --output [Labels]`
* **Export:** `python convert.py onnx`
* **Check:** `python visualize_labels.py --images [Bilder] --labels [Labels] --output [Check]`

Ausführliche Dokumentation siehe: [Prototype and training guide.md](Prototype%20and%20training%20guide.md)
