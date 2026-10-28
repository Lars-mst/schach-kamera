# Figurennetz trainieren

Das Netz in `src/model/` (für die App) entsteht so:

1. Rohdaten nach `rohdaten/` laden (alle von Hugging Face, Lizenzen im Haupt-README):
   - `dopaul/chess-pieces-merged`: `data/*.parquet` als `merged_<split>.parquet`
   - `Francesco/chess-pieces-mjzgj`: `dataset.tar.gz` nach `rf/` entpacken (Ordner train/valid/test)
   - `acapitani/chesspiece-detection-yolo`: `dataset.tar.gz` nach `aca/` entpacken
2. Ausschnitte erzeugen (wie die App sie sieht: Feld plus eine Feldbreite nach oben, 32×64 Pixel,
   dazu leere Felder, auch direkt vor Figuren):
   `RAW=rohdaten DATA=daten python ausschnitte.py dom|rf|aca train|valid|test`
3. Trainieren (CPU reicht, ca. 30–60 Minuten):
   `DATA=daten python training.py dom,rf,aca 14 netz.pt`
   Weitertrainieren mit `INIT=netz.pt LR=1.5e-3`.
4. Exportieren: `python export.py netz.pt ../../src/model`

Abhängigkeiten: `torch numpy pandas pyarrow pillow`.
