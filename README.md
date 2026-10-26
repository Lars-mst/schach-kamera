# Schach Kamera

Browser-App: Kamera aufs Schachbrett richten, die App erkennt das Brett, verfolgt die Züge
und zeigt den besten nächsten Zug (Stockfish) als Overlay direkt auf dem Brett.

## So funktioniert es

1. **Brett finden** (OpenCV.js): größte viereckige Kontur im Bild, danach wird das 8×8-Raster
   so eingepasst, dass die mittleren Felder das klarste Schachmuster zeigen (ein Rand mit
   Koordinaten wird dadurch ignoriert).
2. **Ausrichtung**: a1 ist ein dunkles Eckfeld, Weiß steht auf der Seite mit den helleren Figuren.
   Falls das nicht passt: „Brett drehen“.
3. **Züge erkennen**: Jedes Feld wird mit dem letzten ruhigen Bild verglichen. Sobald sich nichts
   mehr bewegt (Hand weg), wird unter allen legalen Zügen (chess.js) derjenige gewählt, der zu
   den veränderten Feldern passt. Rochade und en passant werden berücksichtigt, Umwandlung
   immer in eine Dame.
4. **Vorschlag**: Stockfish 17.1 (lite, single-threaded WASM) rechnet im Web Worker.

Figuren werden noch nicht einzeln per KI erkannt: Die App startet aus der Grundstellung
(oder einer FEN-Stellung) und leitet jede Figur aus den Zügen ab.

## Lokal starten

```bash
npm install
npm run build
npm run serve   # http://localhost:8080
```

Auf dem iPhone braucht die Kamera HTTPS, daher dort den Vercel-Link nutzen.
Mit `?debug=1` wird oben links die entzerrte Brettansicht eingeblendet.

## Lizenzen

Stockfish.js ist GPLv3, chess.js BSD-2, OpenCV.js Apache 2.0.
