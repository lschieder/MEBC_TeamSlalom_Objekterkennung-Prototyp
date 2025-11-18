import sys
from ultralytics import YOLO

MODEL_PATH = r"C:\TGM 2526 Winter\SEW\MEBC_TeamSlalom_Objekterkennung\runs\detect\train7\weights\best.pt"

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert.py <format>")
        sys.exit(1)

    export_format = sys.argv[1]
    model = YOLO(MODEL_PATH)
    model.export(format=export_format, opset=12, dynamic=True)

if __name__ == "__main__":
    main()
