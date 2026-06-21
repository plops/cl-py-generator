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
*   **Vorteil**: **100% plattformunabhängig und ohne Abhängigkeiten**. Läuft out-of-the-box ohne `libtbb-dev` oder root-Rechte zur Installation. Die Performance ist identisch zur Parallel-STL, da die Last gleichmäßig verteilt wird.
