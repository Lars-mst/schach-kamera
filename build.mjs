// Kopiert die App und die drei Bibliotheken nach dist/ (statische Seite, kein Bundler nötig).
import { cpSync, mkdirSync, rmSync, readdirSync } from 'node:fs';

rmSync('dist', { recursive: true, force: true });
mkdirSync('dist/lib', { recursive: true });
mkdirSync('dist/engine', { recursive: true });

cpSync('src', 'dist', { recursive: true });
cpSync('node_modules/chess.js/dist/esm/chess.js', 'dist/lib/chess.js');
cpSync('node_modules/@techstark/opencv-js/dist/opencv.js', 'dist/lib/opencv.js');

// Stockfish "lite single-threaded": ca. 7 MB, braucht keine COOP/COEP-Header, läuft auf iOS Safari.
const sf = 'node_modules/stockfish/src/';
const base = readdirSync(sf).find(f => /^stockfish-[\d.]+-lite-single-[0-9a-f]+\.js$/.test(f)).replace(/\.js$/, '');
cpSync(sf + base + '.js', 'dist/engine/stockfish.js');
cpSync(sf + base + '.wasm', 'dist/engine/stockfish.wasm');
console.log('dist/ gebaut mit', base);
