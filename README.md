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
6. **Stellung scannen**: Mit gelerntem Figurenset liest die App eine Stellung mitten in der Partie
   ein (Farben aus dem Farbmodell, Figurenart aus den Vorlagen plus Regeln wie Figurenanzahl,
   Läuferfarben und Bauernlinien). Die erkannte Stellung wird zur Bestätigung angezeigt und lässt
   sich per Antippen korrigieren.
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
Mit `?debug=1` wird oben links die entzerrte Brettansicht eingeblendet.

## Lizenzen

Stockfish.js ist GPLv3, chess.js BSD-2, OpenCV.js Apache 2.0.
