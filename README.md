# Schach Kamera

Browser-App: Kamera aufs Schachbrett richten, die App erkennt das Brett, verfolgt die Züge
und zeigt den besten nächsten Zug (Stockfish) als Overlay direkt auf dem Brett.

## So funktioniert es

1. **Brett finden** (OpenCV.js): größte viereckige Kontur im Bild, danach wird das 8×8-Raster
   so eingepasst, dass die mittleren Felder das klarste Schachmuster zeigen (ein Rand mit
   Koordinaten wird dadurch ignoriert).
2. **Ausrichtung**: a1 ist ein dunkles Eckfeld, Weiß steht auf der Seite mit den helleren Figuren.
   Falls das nicht passt: „Brett drehen“. Die eigene Farbe ist die Seite, die näher an der Kamera
   liegt (größere Felder, weiter unten im Bild); steht das Brett quer, gibt es Tipps für beide Seiten.
   Stammt die Stellung aus einem Scan, korrigiert die Zugerkennung Figurenarten anhand der Gangart
   (z. B. ein „Bauer“, der schräg über drei Felder zieht, ist ein Läufer).
3. **Züge erkennen**: Jedes Feld wird mit dem letzten ruhigen Bild verglichen. Sobald sich nichts
   mehr bewegt (Hand weg), wird unter allen legalen Zügen (chess.js) derjenige gewählt, der zu
   den veränderten Feldern passt. Rochade und en passant werden berücksichtigt, Umwandlung
   immer in eine Dame.
4. **Vorschlag**: Stockfish 17.1 (lite, single-threaded WASM) rechnet im Web Worker.

5. **Abgleich mit dem Bild**: Aus jeder bestätigten Stellung lernt die App, wie leere Felder,
   weiße und schwarze Figuren bei deinem Licht aussehen (Farbmodell je Feldfarbe), und speichert
   kleine Bildvorlagen jeder Figurenart (im Browser, `localStorage`). Passt das Bild nicht mehr zur
   Stellung, sucht sie die Zugfolge (bis zu 3 Halbzüge, auch mit Korrektur des letzten Zuges), die
   das Bild erklärt. Findet sie keine, gibt es keinen Vorschlag, sondern einen Hinweis zum Neu-Einlesen.
6. **Stellung einlesen mit dem Figurennetz**: Ein kleines, selbst trainiertes Faltungsnetz
   (`src/piecenet.js`, Gewichte in `src/model/`, ca. 330 KB) bestimmt für jedes Feld „leer“ oder eine
   der 12 Figuren. Es läuft in reinem JavaScript in einem Web Worker, also ohne Zusatzbibliothek und
   auch auf dem iPhone. Damit liest die App jede Stellung sofort ein, ohne vorher etwas lernen zu
   müssen. Die Regeln (genau ein König je Seite, Figurenanzahl, Läuferfarben, Bauernlinien) legen
   die endgültige Verteilung fest. Aus jeder sicher verfolgten Stellung merkt sich die App außerdem
   die Netz-Merkmale deines Figurensets und gleicht neue Bilder zusätzlich damit ab (Anpassung an
   Figuren, Brett und Licht). Stammt die Stellung aus einem Scan, prüft das Netz in Ruhephasen
   laufend alle Figuren und korrigiert eine Figurenart, wenn es mehrmals hintereinander sehr deutlich
   widerspricht (nie den König). Ohne Netz (z. B. Laden fehlgeschlagen) gilt das alte Verfahren mit
   Farbmodell und Bildvorlagen.
7. **Nachführen**: Bewegt sich die Kamera, wird der Brettumriss laufend neu gesucht, die
   Ausrichtung bleibt erhalten. Jede Figur bekommt einen perspektivischen Ring auf ihrem Feld.

## Lokal starten

```bash
npm install
npm run build
npm run serve   # http://localhost:8080
```

Auf dem iPhone braucht die Kamera HTTPS, daher dort https://lars-mst.github.io/schach-kamera/ nutzen.
Jeder Push auf `main` baut die App neu und veröffentlicht sie über den Zweig `gh-pages`.
Mit `?debug=1` wird oben links die entzerrte Brettansicht eingeblendet. In den Einstellungen gibt es
außerdem eine Diagnoseansicht (Raster, Belegung je Feld in Prozent, Ecken, FEN, Engine-Variante),
optionale Zusatzpfeile für die zweit- und drittbeste Variante (Stockfish MultiPV), PGN-Export und
einen Link, der die Stellung in der Lichess-Analyse öffnet.

Fair Play: Die App ist für Training und Analyse gedacht, nicht für Partien, in denen Computerhilfe
verboten ist.

## Figurennetz

Training: `tools/figurennetz/` (PyTorch, eigener Code). Aus Detektionsdatensätzen mit
Figuren-Rahmen werden Feldausschnitte erzeugt, wie die App sie sieht (Feld plus eine Feldbreite nach
oben, 32×64 Pixel), dazu Ausschnitte leerer Felder. Das Netz hat sechs 3×3-Faltungsschichten und
13 Ausgänge. Die Gewichte werden mit eingerechneter BatchNorm als Float32 exportiert.

Trainingsdaten (alle mit freier Lizenz, Quellenangabe nach CC BY 4.0):

- „Chess Pieces“ von Roboflow (Roboflow 100, `chess-pieces-mjzgj`), CC BY 4.0,
  über https://huggingface.co/datasets/Francesco/chess-pieces-mjzgj
- „chess_pieces_dominique“, CC BY 4.0, über https://huggingface.co/datasets/dopaul/chess-pieces-merged
- „Chess Pieces Detection (YOLO)“ von acapitani, MIT, https://huggingface.co/datasets/acapitani/chesspiece-detection-yolo,
  enthält Bilder aus ChessReD (Masouris & van Gemert, „End-to-End Chess Recognition“, 2024), CC BY 4.0

Es wurden keine fertigen YOLO-/Ultralytics-Gewichte (AGPL) und kein GPL-/AGPL-Code übernommen.

## Lizenzen

Stockfish.js ist GPLv3, chess.js BSD-2, OpenCV.js Apache 2.0. Das Figurennetz ist selbst trainiert
(Daten siehe oben).
