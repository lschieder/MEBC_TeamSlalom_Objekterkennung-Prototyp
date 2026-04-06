

# Prototyp: MEBC - TeamSlalom - Objekterkennung

Autor: Laurin Schieder

Version: 28.10.2025



### Neues Python-Environment anlegen und aktivieren

conda create --name yolo11-env python=3.12 -y
conda activate yolo11-env

### Ultralytics und PyTorch installieren

```bash
pip install ultralytics
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

```

Checken ob Pytorch GPU richtig installiet ist:

```bash
python -c "import torch; print(torch.cude.get_device_name(0))"
```

![](C:\Users\lauri\AppData\Roaming\marktext\images\2025-10-28-11-20-03-image.png)

### Daten sammeln & labeln

### Labels erstellen

Zb. Mit Labelstudio oder **Auto-Labeling** (siehe unten).

### Datenstruktur vorbereiten

Quelle: `curl --output train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py`

Teilt einen Datensatz in `train` und `validation` Ordner auf.

```bash
# --datapath: Ordner der 'images' und 'labels' enthält
python train_val_split.py --datapath="C:\Pfad\Zu\Daten" --train_pct=.8
```

### data.yaml für das Training anlegen

Öffne Texteditor, erstelle und passe diese Datei an:

```yml
path: C:\Objectdetection\Objectdetection\data
train: train\images
val: validation\images
nc: 2
names: ["EndBoje", "KursBoje"]
```



### Training starten (mit GPU)

Stelle sicher, dass PyTorch mit CUDA installiert ist (siehe oben).

```python
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```

Das Modell wird dann als pt datei gespeichert bzw die weights unter `Objectdetection\runs\detect\train\weigts`

---

## Hilfsskripte & Tools

### yolo_detect.py
Führt die Objekterkennung live auf Videoquellen oder Dateien aus.
*Quelle: Basiert auf dem Repository aus dem oben genannten `train_val_split.py` Guide.*

```bash
# Beispiel: Erkennung auf OBS Virtual Camera
python yolo_detect.py --model runs/detect/train2/weights/best.pt --source usb0

# Beispiel: Erkennung auf Videodatei
python yolo_detect.py --model runs/detect/train2/weights/best.pt --source mein_video.mp4 --output ergebnis.avi
```

### convert.py
Konvertiert ein trainiertes PyTorch Modell (`.pt`) in andere Formate (z.B. ONNX für Web/Mobile).

```bash
python convert.py onnx
```

### visualize_labels.py
Zeichnet zur Kontrolle Bounding-Boxes aus den `.txt` Files direkt in die Bilder ein.

```bash
python visualize_labels.py --images "Pfad/Bilder" --labels "Pfad/Labels" --output "Pfad/Check"
```

### test.py
Einfaches Skript zur Überprüfung der PyTorch und CUDA Installation.

```bash
python test.py
```

### auto_label.py
Nutzt ein existierendes Modell, um ungelabelte Bilder automatisch vorzulabeln.

```bash
python auto_label.py --model runs/detect/trainX/weights/best.pt --source "Bilder" --output "NeueLabels"
```

---

### Analyse der Ergebnise

![](C:\Users\lauri\AppData\Roaming\marktext\images\2025-10-28-16-35-20-confusion_matrix_normalized.png)

Die normalisierte Confusion Matrix zeigt die Leistung eines Klassifikationsmodells für drei Klassen (EndBoje, KursBoje, background), wobei EndBoje perfekt erkannt wird (1.00), KursBoje sehr gut klassifiziert wird (0.96), aber beide Bojen-Klassen zu 50% mit dem Hintergrund verwechselt werden. Das Modell hat Schwierigkeiten bei der Unterscheidung zwischen den Objektklassen und dem Hintergrund, während die Unterscheidung zwischen EndBoje und KursBoje nahezu fehlerfrei funktioniert.



![](C:\Users\lauri\AppData\Roaming\marktext\images\2025-10-28-16-41-46-results.png)



Die Grafik zeigt die Trainingsverläufe eines YOLO-Objekterkennungsmodells über etwa 60 Epochen, wobei alle Loss-Metriken (box_loss, cls_loss, dfl_loss) sowohl für Training als auch Validierung kontinuierlich sinken und sich stabilisieren. Die Evaluationsmetriken (Precision, Recall, mAP50, mAP50-95) steigen während des Trainings konstant an und erreichen gegen Ende Werte nahe 1.0 für Precision und Recall sowie etwa 0.95-1.0 für mAP50, was auf ein erfolgreiches Training mit guter Konvergenz hinweis
