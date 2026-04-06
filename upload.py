from roboflow import Roboflow

# --- Skript zum Hochladen eines Datensatzes zu Roboflow ---

# Roboflow Client mit API-Key initialisieren
rf = Roboflow(api_key="******")
# Workspace festlegen
workspace = rf.workspace("slalom-tdduk")

# Datensatz hochladen (es wird automatisch nach Bildern und .txt Label-Dateien gesucht)
workspace.upload_dataset(
    dataset_path=r"E:\opt_dataset\unity",  # Pfad zum lokalen Datensatz
    project_name="Dataset",  # Zielprojekt in Roboflow
    num_workers=1,  # Anzahl der parallelen Uploads
    project_type="object-detection",  # Typ des Projekts
    batch_name="unity",  # Name des Upload-Batches zur Identifizierung in Roboflow
    num_retries=3  # Versuche bei Verbindungsabbruch
)
