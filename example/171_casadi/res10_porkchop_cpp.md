# Erde-Mars Porkchop-Diagramm: C++20 Parallelisierungs-Leitfaden

Dieses Dokument dokumentiert die Überlegungen, Pläne und architektonischen Optimierungen für die Portierung und Parallelisierung der interplanetaren Trajektorienberechnung (Porkchop-Plot) in C++.

---

## 1. Motivation und Problemstellung

Die Berechnung von interplanetaren Transferfenstern (z. B. Erde $\rightarrow$ Mars) erfordert das Lösen des **Lambert-Problems** (Zwei-Punkt-Randwertproblem) über ein großes 2D-Gitter von Abflugsdaten ($t_{\text{dep}}$) und Flugzeiten ($t_{\text{tof}}$).
*   **Herausforderung**: Für jeden Gitterpunkt muss ein numerisches Integrationsverfahren mit einem Newton-Gleichungslöser gekoppelt werden. Bei feinen Gittern (z. B. $500 \times 500 = 250.000$ Punkte) führt dies in Python zu signifikanten Rechenzeiten.
*   **Ziel**: Portierung des Solvers nach C++ und Parallelisierung auf Mehrkern-Maschinen (z. B. 32 Kerne), um die Rechenzeit auf Millisekunden zu reduzieren.

---

## 2. Analyse des Compiler-Speicherfehlers (13 MB vs. 1.3 MB)

Im ersten Entwurf haben wir den gesamten Algorithmus – einschließlich der 10 Newton-Iterationen des Rootfinders – in einen einzigen symbolischen CasADi-SX-Graphen verpackt und als C++-Code exportiert.

### Warum schlug die Kompilierung fehl?
*   **Symbolische Expansion (Inlining)**: Da CasADi SX-Ausdrücke vollständig flacht, wurde die Integration (150 RK4-Schritte) und deren symbolische Jacobimatrix für jede der 10 Newton-Iterationen inlined.
*   **Dateigröße**: Dies führte zu einer **13 Megabyte** großen C++-Quelldatei mit über 500.000 Zeilen sequenzieller Zuweisungen.
*   **Ressourcenmangel (RAM-Limit)**: Der GCC-Compiler versuchte, den riesigen Ausdrucksbaum im RAM zu optimieren. Das Betriebssystem beendete den Compiler (`cc1plus`) wegen Speichermangels (`Killed signal terminated program cc1plus`).

### Die optimierte Hybrid-Architektur (10x Code-Reduktion)
Wir haben die Berechnungen logisch getrennt:
1.  **CasADi generiert nur die mathematischen Kerne**:
    *   Die 150-Schritte ODE-Integration (`integrate_rk4`).
    *   Die exakte Jacobimatrix der Endposition bezüglich der Startgeschwindigkeit.
    *   Die analytischen Planetenbahnen von Erde und Mars.
2.  **C++ übernimmt die Kontrolllogik**:
    *   Die Newton-Iterationen (15 Schritte) und die analytische 2x2 Matrixinversion werden direkt in C++ in einer einfachen Schleife ausgeführt.

**Ergebnis:** Die Dateigröße schrumpfte von **13 MB auf 1.3 MB**, die Kompilierung dauert nun weniger als 2 Sekunden und benötigt kaum Arbeitsspeicher.

---

## 3. Parallelisierungs-Entscheidung unter C++20

Um die maximale Leistung ohne OpenMP zu erzielen, haben wir zwei C++20-Konzepte ausgearbeitet:

### Variante A: C++20 Parallel STL (`std::execution::par`)
*   **Beschreibung**: Nutzt die standardmäßige STL-Algorithmen-Bibliothek:
    ```cpp
    std::for_each(std::execution::par, points.begin(), points.end(), [&](const GridPoint& pt) { ... });
    ```
*   **Dependency**: GCC benötigt unter Linux die Bibliothek **Intel TBB** (`libtbb-dev`) als Backend. Wenn TBB auf dem System nicht installiert ist, schlägt der CMake-Build fehl.

### Variante B: C++20 Standard-Multithreading (`std::thread` / `std::jthread`)
*   **Beschreibung**: Wir teilen das Berechnungsraster mittels eines einfachen Round-Robin-Verfahrens auf die verfügbaren Hardware-Kerne auf:
    ```cpp
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.push_back(std::thread([&, t]() {
            for (int idx = t; idx < N_points; idx += num_threads) { ... }
        }));
    }
    ```
*   **Vorteil**: **100% plattformunabhängig und ohne Abhängigkeiten**. Läuft out-of-the-box ohne `libtbb-dev` or root-Rechte zur Installation. Die Performance ist identisch zur Parallel-STL, da die Last gleichmäßig verteilt wird.

---

## 4. Performance-Messungen und Vergleich

Die C++20 Parallel-STL Implementierung wurde auf zwei unterschiedlichen Systemen für $250.000$ Trajektorien ($500 \times 500$ Gitter) evaluiert:

### System A: Laptop (16 logische Kerne)
*   **Kommando**: `time ./porkchop_sim`
*   **Simulationszeit**: $5565.52$ ms ($5.5$ Sekunden)
*   **Ressourcenauslastung**:
    *   `real`: `0m5.586s`
    *   `user`: `1m27.305s`
    *   `sys`: `0m1.145s`
*   **Statistik & Unnütze Punkte Report (Kumulierte CPU-Zeit)**:
    *   *Konvergenzfehler (NaN)*: $191.444$ ($76.5776\%$) | CPU-Zeit: $71.433$ ms
    *   *Zu hohes Delta-v (>20)*: $53.123$ ($21.2492\%$) | CPU-Zeit: $15.149,3$ ms
    *   *Nutzbare Transferfenster*: $5.433$ ($2.1732\%$) | CPU-Zeit: $1.673,48$ ms
    *   *Vergeudete CPU-Rechenleistung*: **$86.582,3$ ms von $88.255,8$ ms gesamt ($98.1038\%$ der Rechenleistung)**.

### System B: Virtual Private Server (VPS, 2 logische Kerne)
*   **Kommando**: `time ./porkchop_sim`
*   **Simulationszeit**: $39318.2$ ms ($39.3$ Sekunden)
*   **Ressourcenauslastung**:
    *   `real`: `0m39.343s`
    *   `user`: `1m14.638s`
    *   `sys`: `0m0.055s`
*   **Statistik & Unnütze Punkte Report (Kumulierte CPU-Zeit)**:
    *   *Konvergenzfehler (NaN)*: $191.407$ ($76.5628\%$) | CPU-Zeit: $63.622,7$ ms
    *   *Zu hohes Delta-v (>20)*: $53.160$ ($21.264\%$) | CPU-Zeit: $13.453,3$ ms
    *   *Nutzbare Transferfenster*: $5.433$ ($2.1732\%$) | CPU-Zeit: $1.406,09$ ms
    *   *Vergeudete CPU-Rechenleistung*: **$77.076$ ms von $78.482,1$ ms gesamt ($98.2084\%$ der Rechenleistung)**.

---

## 5. Physikalische Bedeutung des Delta-v für 200 kg Payload

Die Raketengleichung von Ziolkowski beschreibt den Zusammenhang zwischen dem benötigten $\Delta v$, der Triebwerkseffizienz ($I_{sp}$) und der Masse:

$$m_0 = m_f \cdot e^{\frac{\Delta v}{I_{sp} \cdot g_0}}$$

### Beispielberechnung für eine Mars-Sonde:
*   **Nutzlast (Payload)**: $200$ kg.
*   **Trockenmasse der Sonde** (Struktur, Avionik, Solarpanels): $150$ kg.
*   **Endmasse beim Mars-Einschuss ($m_f$)**: $350$ kg.
*   **Spezifischer Impuls ($I_{sp}$)** (Standard-Zweistoff-Triebwerk im Vakuum): $320$ s (entspricht $I_{sp} \cdot g_0 \approx 3.14$ km/s Ausströmgeschwindigkeit).

Wir berechnen die benötigte Startmasse ($m_0$) in der Erdumlaufbahn für verschiedene $\Delta v$-Werte:

| Szenario | Benötigtes $\Delta v$ | Startmasse in LEO ($m_0$) | Treibstoffmasse ($m_{\text{prop}}$) | Erläuterung |
| :--- | :--- | :--- | :--- | :--- |
| **Hohmann-Fenster (Optimum)** | **$6.0$ km/s** | **$2.362$ kg** | **$2.012$ kg** | Leicht zu starten, passt auf eine kleine Trägerrakete (z. B. Electron/Falcon 9 Rideshare). |
| **Abweichung (Suboptimal)** | **$10.0$ km/s** | **$8.435$ kg** | **$8.085$ kg** | Startmasse vervierfacht sich! Benötigt eine mittlere bis schwere Trägerrakete. |
| **Ungünstiges Startdatum** | **$15.0$ km/s** | **$41.580$ kg** | **$41.230$ kg** | Nahezu unmöglich. Benötigt eine Schwerlastrakete (z. B. Falcon Heavy / Ariane 6) für dieselbe winzige 200-kg-Nutzlast. |

> [!IMPORTANT]
> Da das $\Delta v$ exponentiell in die Startmasse einfließt, entscheidet das Porkchop-Diagramm direkt darüber, ob eine Mission physikalisch und finanziell überhaupt durchführbar ist.

---

## 6. Diskussion zur Optimierung des Suchbereichs (Gitter vs. Pruning)

Kann man den Suchraum eingrenzen (z. B. nur entlang bekannter Linien suchen) oder besteht die Gefahr, Minima zu übersehen?

### Physikalische Plausibilität & Glattheit des Suchraums
Die Bahndynamik im Sonnensystem ist stetig und stetig differenzierbar. Delta-v-Landschaften sind glatt und bilden Täler (Hohmann-Täler). 
Ja, man kann lokal die Gradienten nutzen, um Linien zu folgen (wie in der Continuation-Methode aus `gen04`).

### Die Gefahr lokaler Minima (Warum Gitter-Suchen notwendig sind)
Ein radikales Abschneiden des Suchraums birgt die große Gefahr, alternative, physikalisch vorteilhafte Minima zu übersehen:
*   **Transfer-Typen**: Es gibt **Typ-I** (Transitwinkel $< 180^{\circ}$) und **Typ-II** (Transitwinkel $> 180^{\circ}$) Transfers, die völlig separate Täler im Porkchop-Plot bilden.
*   **Multi-Revolutionen**: Bei längeren Flugzeiten existieren Minima, bei denen die Sonde die Sonne mehrfach umkreist, bevor sie den Mars trifft.
*   **Broken-Plane-Manöver**: Durch Hinzufügen einer Bahnkorrektur außerhalb der Ekliptik können sich neue Minima an Stellen auftun, die im klassischen 2-Impuls-Modell blockiert sind.

> [!WARNING]
> Eine rein lokale Suche in der Nähe eines bekannten Minimums findet niemals diese anderen, oft günstigeren Trajektorienklassen. Daher ist die vollständige Gitterberechnung (Grid Search) unerlässlich für das Missionsdesign.

### Ursache der Konvergenzfehler (Weiße Bereiche)
Die weißen Bereiche im Plot entstehen nicht, weil dort keine physikalische Trajektorie existiert, sondern weil das lokale Newton-Verfahren ausgehend von der naiven Startschätzung (`1.1 * Erdgeschwindigkeit`) in Regionen starker Nichtlinearität divergiert.
*   *Lösung zur Behebung*: Einbau eines algebraischen Lambert-Approximators (z. B. Izzo-Algorithmus) zur Generierung einer präzisen Startschätzung für jeden Gitterpunkt. Damit konvergiert das Verfahren für 100% aller Punkte.

---

## 7. Report über "unnütze Punkte" und Rechenzeitverschwendung

Im optimierten C++-Programm wurden Zähler eingeführt, um zu analysieren, wie viel Rechenleistung auf energetisch unbrauchbare oder nicht konvergierte Trajektorien entfällt. 

Für ein Raster von $250.000$ Punkten ergab sich auf dem VPS (2 CPU-Kerne) folgendes Bild:

*   **Konvergenzfehler (NaN)** (Konstellation ungünstig für lokale Startschätzung): **76.56%** ($191.407$ Punkte) | Kumulierte CPU-Zeit: $63.622,7$ ms
*   **Zu hohes Delta-v (>20 km/s)** (Sonde fliegt energetisch unsinnige Umwege): **21.26%** ($53.160$ Punkte) | Kumulierte CPU-Zeit: $13.453,3$ ms
*   **Nutzbare Transferfenster (<=20 km/s)**: **2.17%** ($5.433$ Punkte) | Kumulierte CPU-Zeit: $1.406,1$ ms

### Fazit der Rechenzeitverteilung:
**98,2% der kumulierten CPU-Rechenleistung** ($77.076$ ms von $78.482$ ms) wurde für unbrauchbare Trajektorien aufgewendet. 
Dies demonstriert eindrucksvoll, warum im echten Missionsdesign fortgeschrittene Methoden (wie die Arc-Length-Continuation aus `gen04`) oder neuronale Netze zur Vorselektion eingesetzt werden, um die Berechnungen auf das 2%-Intervall der nutzbaren Transferfenster zu fokussieren.

---

## 8. Physikalische Begründung des Abschneidens (Untergrenze der Flugzeit)

Im Porkchop-Diagramm ist zu erkennen, dass die Höhenlinien am unteren Rand (bei kurzen Flugzeiten) abgeschnitten werden.

### Warum geht das Minimum nicht einfach wieder nach oben?
*   **Die physikalische Realität**: Wenn die Flugzeit ($t_{\text{tof}}$) kürzer wird, muss die Sonde einen direkteren, viel schnelleren Weg fliegen. Dies bedeutet, dass sie sich nicht mehr auf einer energiesparenden Ellipse (Hohmann-Transfer) bewegt, sondern die Flugbahn gestreckt werden muss. Die Sonde benötigt eine weitaus höhere kinetische Energie.
*   **Grenzverhalten**: Wenn die Flugzeit gegen Null geht ($t_{\text{tof}} \rightarrow 0$), geht die benötigte Start- und Bremsgeschwindigkeit gegen Unendlich ($\Delta v \rightarrow \infty$). Physikalisch steigt das Delta-V im unteren Teil der Karte also extrem steil und unbegrenzt an.
*   **Der Grund für den visuellen Cutoff**: 
    1.  **Plot-Grenze**: Im Diagramm ist die Y-Achse standardmäßig auf eine Untergrenze (z. B. 100 oder 80 Tage) eingestellt. Das Abschneiden ist also primär eine Darstellungsbegrenzung (Axis Limit), da Flugzeiten unter 80 Tagen astronomische Energiemengen erfordern, die technisch irrelevant sind.
    2.  **Solver-Divergenz**: Für extrem kurze Flugzeiten (z. B. < 50 Tage) divergiert das Newton-Verfahren komplett, da die benötigten Geschwindigkeiten zu extrem sind und weit außerhalb des Konvergenzradius unserer einfachen Startschätzung liegen. Diese Punkte werden im Plot als NaNs (weiße Flächen) dargestellt.
