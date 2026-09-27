Dass Plotly auf Smartphones versagt, hat handfeste architektonische Gründe:

1. **4,8 MB JS-Bundle:** Mobilfunk-Latenz und JavaScript-Parse-Time auf mobilen SOCs bremsen die Seite massiv aus.
2. **Das „Fat-Finger“-Problem:** Plotly setzt auf pixelgenaue Maus-Events (`hover` existiert auf Touchscreens schlichtweg nicht). Bei 16.692 dicht beieinander liegenden Punkten mit dem Daumen einen bestimmten Punkt zu treffen, führt fast immer zu Fehlklicks.
3. **220 Traces in der Legende:** Eine Liste mit 220 Einträgen zerstört jedes mobile Layout.
4. **Desktop-Sidebar (360 px):** Auf einem Smartphone mit z. B. 390 px Viewport-Breite (iPhone) verdeckt ein 360 px Seiten-Panel die Karte komplett.

Da Ihr Rust/Axum-Backend (`/api/map/points`, `/api/map/clusters`, `/api/map/point/{id}`) exzellent und schlank designt ist, muss am Rust-Code **nichts** geändert werden. Der Wechsel betrifft rein das Askama-Template `templates/map.html`.

Hier sind die drei besten Alternativ-Implementierungen, geordnet nach Einsatzzweck, gefolgt von den zwingend notwendigen UI-Anpassungen für Touch-Bedienung.

---

### Alternative 1: Leaflet mit `L.CRS.Simple` + Canvas / WebGL (Die pragmatische Empfehlung)

Leaflet ist seit über einem Jahrzehnt der Goldstandard für mobile Karten. Touch-Gesten (Pinch-to-Zoom, Trägheits-Pan, Doppeltipp) laufen nirgends so flüssig und out-of-the-box wie hier. 

Obwohl Leaflet für Weltkarten gedacht ist, unterstützt es über `L.CRS.Simple` planare kartesische Koordinaten ($X, Y$) für zweidimensionale Cluster-/Embedding-Karten (UMAP/t-SNE).

* **Bundle-Größe:** ca. **140 KB** (Leaflet) + **15 KB** Plugin statt 4.800 KB Plotly.
* **Rendering von 16.700 Punkten:** DOM-Marker würden crashen. Die Lösung ist ein Canvas-Overlay:
  * Entweder **Leaflet.Canvas-Markers** (reines 2D-Canvas, reicht für ~17k Punkte auf modernen Handys völlig aus).
  * Oder **Leaflet.glify** (WebGL-beschleunigt, rendert 100k+ Punkte mit 60 FPS).
* **Vorteil:** Native Touch-Gestik, extrem leichtgewichtig, keine Canvas-Zoom-Mathematik per Hand nötig.

```javascript
// Minimales Setup für 2D-Scatter:
const map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -5,
    maxZoom: 3,
    zoomSnap: 0.25,
    attributionControl: false
});

// Koordinaten zentrieren
map.setView([centerY, centerX], 0);

// Punkte via L.glify (WebGL) oder Canvas-Renderer zeichnen:
L.glify.points({
    map: map,
    data: pointsData, // [[y, x], ...]
    size: 6,
    color: (i, point) => clusterColors[point.cluster_id],
    click: (e, point) => loadPointDetails(point.id)
});
```

---

### Alternative 2: Reines HTML5 Canvas 2D + `d3-zoom` + `kdbush` (Das Ultra-Lightweight-Leichtgewicht)

Wenn Sie gar kein großes Karten-Framework einbinden wollen, ist eine eigene kleine Canvas-Lösung oft die robusteste Wahl. 16.692 Punkte sind für ein HTML5 `CanvasRenderingContext2D` auf modernen Smartphones ein Kinderspiel (Renderzeit < 10 ms).

* **Bundle-Größe:** ca. **25 KB** gesamt!
  * `d3-zoom` (~15 KB gzipped): Übernimmt Pinch-to-Zoom und Pan perfekt für Touch & Desktop.
  * `kdbush` / `flatbush` (~3 KB): Ein winziger 2D-Spatial-Index im Browser.
* **Wie es das Mobile-Problem löst:**
  Wenn der Nutzer auf das Display tippt, feuert ein `pointerup`. Dank `kdbush` fragen Sie in 0,1 Millisekunden ab: *„Welcher Punkt liegt innerhalb von 20 Pixeln um den Daumen?“* Liegen mehrere Punkte im Radius, wählen Sie den nächstgelegenen. Das verhindert Fehlklicks vollständig.
* **Vorteil:** Volle Kontrolle über das Rendering, keine Abhängigkeiten von Riesen-Bibliotheken, lädt instant im Mobilfunk.

---

### Alternative 3: Deck.gl (`ScatterplotLayer`) oder Cosmograph

Wenn Sie zwingend GPU-beschleunigte Visualisierungen wollen (z. B. wenn die Datenmenge künftig auf 100.000+ Punkte wächst):

* **Cosmograph:** Speziell für Embeddings, semantische Netzwerke und t-SNE/UMAP-Cluster gebaut. Bietet fertige Interaktions-Mechaniken.
* **Deck.gl (vis.gl):** Extrem performant, unterstützt integriertes Raycasting für Touch-Picking (`pickingRadius: 15` für Daumen-Toleranz).
* **Nachteil:** Größere Bundles (~500 KB–1 MB) und höhere Einarbeitungszeit als Leaflet oder 2D Canvas.

---

### Unverzichtbar: UI/UX-Umbau für Mobile (egal welche Library!)

Plotlys Bedienung auf Mobilgeräten scheitert nicht nur an der Render-Engine, sondern am Layout-Konzept. Für Smartphones sollten Sie folgende 3 Punkte im Askama-Template umsetzen:

#### 1. Bottom Sheet statt 360 px Sidebar
Auf Mobilgeräten darf es keine feste Sidebar geben. Nutzen Sie stattdessen ein **Bottom Sheet** (Slide-up Drawer):
* **Zustand A (Standard):** Karte füllt 100% des Screens.
* **Zustand B (Punkt angetippt):** Ein kleines Panel schiebt sich von unten 80 px hoch (zeigt Cluster-Name + Video-Titel).
* **Zustand C (Volltext lesen):** Der Nutzer zieht das Panel nach oben (oder tippt darauf) – das Sheet expandiert auf 80–90% der Bildschirmhöhe und zeigt die Zusammenfassung sowie den YouTube-Link.

#### 2. Legende in Modal/Drawer auslagern
Eine Liste mit 220 Cluster-Einträgen funktioniert mobil nicht als dauerhafte Legende.
* **Lösung:** Ein schwebender runder Button unten rechts: `[ ☰ 219 Cluster ]`.
* Ein Klick darauf öffnet ein durchsuchbares Overlay/Modal (`<input type="search" placeholder="Cluster suchen...">`).
* Dort kann der Nutzer Cluster an-/abwählen oder gezielt einen Cluster fokussieren („Kamera zoomt auf Cluster 42“).

#### 3. Touch-Hitbox (Bounding Circle) statt Hover
* Ersetzen Sie `hover` durch `tap` (bzw. `pointerdown` / `pointerup`).
* Legen Sie einen virtuellen Klick-Radius von ca. **16–24 px** um die Berührstelle. Ermitteln Sie über eine einfache Euklidische Distanz (`(x - x0)² + (y - y0)² < r²`) den Punkt, der dem Finger am nächsten liegt.

---

### Empfohlener Fahrplan für Ihr Projekt

1. **Askama Template anpassen:** Entfernen Sie Plotly (`cdn.plot.ly/...`) aus `templates/map.html`.
2. **Leaflet + `L.CRS.Simple` + Canvas-Marker einbinden:**
   * CSS und JS von Leaflet wiegen zusammen < 150 KB.
   * Lassen Sie die Endpunkte `/api/map/points` und `/api/map/clusters` unverändert. Mappen Sie die JSON-Koordinaten einfach auf `[y, x]`.
3. **Pico-CSS-kompatibles Bottom-Sheet implementieren:** Ein einfaches CSS-Panel, das per Klasse `.active` von `bottom: -100%` auf `bottom: 0` translatiert.
4. **Cluster-Filter:** Ein simples `<dialog>`-Element (nativ in HTML5), das die 220 Cluster als scrollbare Liste mit Checkboxen und Suchleiste kapselt.

Mit dieser Kombination sinkt die Seitenladezeit auf Mobilgeräten von spürbaren Sekunden auf wenige Millisekunden, und die Karte lässt sich flüssig wie Google Maps mit zwei Fingern bedienen.
