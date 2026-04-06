import sys
from ultralytics import YOLO

# Pfad zum trainierten YOLO-Modell (.pt-Datei)
# Muss angepasst werden, wenn ein anderes Training (z.B. train2, train3) exportiert werden soll.
MODEL_PATH = r"C:\Objectdetection\Objectdetection\runs\detect\train\weights\best.pt"

def main():
    """
    Konvertiert ein YOLO PyTorch Modell (.pt) in ein anderes Format.
    Verwendung: python convert.py <format> (z.B. onnx, tflite, engine)
    """
    if len(sys.argv) < 2:
        print("Usage: python convert.py <format>")
        sys.exit(1)

    export_format = sys.argv[1]

    # Modell laden
    model = YOLO(MODEL_PATH)

    # Export starten
    # opset=12 wird oft für ONNX Kompatibilitaet verwendet
    # nms=True fügt Non-Maximum Suppression direkt in das exportierte Modell ein
    model.export(format=export_format, opset=12, nms=True)

if __name__ == "__main__":
    main()
