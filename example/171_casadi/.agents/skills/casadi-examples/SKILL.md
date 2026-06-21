---
name: casadi-examples
description: Richtlinien für die Erstellung hochqualitativer, physikalischer Simulationsbeispiele in Common Lisp mit CasADi
---

# Richtlinien für CasADi-Simulationsbeispiele

Dieses Dokument enthält die Standards und Anforderungen für die Erstellung von Demonstrations- und Simulationsbeispielen im Projekt. Ein CasADi-Beispiel muss so gestaltet sein, dass es sowohl physikalisch korrekt als auch für Entwickler leicht verständlich und nachvollziehbar ist.

---

## 1. Physikalische Modellierung & Erläuterung
*   **Modellspezifikation**: Jedes physikalische System muss exakt beschrieben werden. Bei Fahrzeugen ist z. B. zu spezifizieren, ob es sich um ein *Viertelfahrzeug (Quarter-Car Model)* handelt (Annahme der Entkopplung der Ecken) oder um ein vereinfachtes Punktmasse-Modell.
*   **Einheiten**: Alle physikalischen Parameter (Massen, Steifigkeiten, Dämpfungskonstanten, Kräfte, Zeiten) müssen mit ihren SI-Einheiten (kg, N/m, N-s/m, N, s) in den Kommentaren deklariert werden.
*   **Zustandsdefinitionen**: Der Zustandsvektor $\mathbf{x}$ und der Steuerungsvektor $\mathbf{u}$ müssen komponentenweise mit physikalischer Bedeutung und Einheit aufgeschrieben werden.

---

## 2. Numerische Stabilität & Diskretisierung
*   **Keine instabilen Integratoren**: Explizite Euler-Diskretisierung führt bei Systemen mit Eigenfrequenzen nahe oder oberhalb des Abtasttheorems (z. B. Reifen-Masse-Schwingungen) schnell zur Instabilität ("explodierende Plots").
*   **Exakte Diskretisierung**: Für lineare Systeme soll die exakte Diskretisierung über das Matrix-Exponential (`scipy.linalg.expm`) verwendet werden:
    $$\mathbf{x}_{k+1} = \mathbf{A}_d \mathbf{x}_k + \mathbf{B}_d \mathbf{u}_k + \mathbf{G}_d \mathbf{v}_{k}$$
    wobei $\mathbf{A}_d = e^{\mathbf{A}_c \Delta t}$ und Eingangsmatrizen über das erweiterte System diskretisiert werden.

---

## 3. Dokumentation von Optimierungsparametern
Alle CasADi-spezifischen Optimierungsparameter und -variablen müssen im Code präzise dokumentiert werden:
*   `lbx` / `ubx`: Untere/obere Grenzen der Optimierungsvariablen (Entscheidungsvariablen wie Zustände und Stellgrößen).
*   `lbg` / `ubg`: Untere/obere Grenzen der Nebenbedingungen (für Systemdynamik-Gleichungen auf $0$ gesetzt, um Gleichheitsnebenbedingungen zu erzwingen).
*   `p`: Parametervektor des Optimierers (z. B. aktueller Zustand $\mathbf{x}_0$ und Störgrößenvorschau).
*   `x0_guess`: Startschätzung für den QP/NLP-Löser.

---

## 4. Kostenfunktionen und Dämpfung
*   **Gewichtungskoeffizienten**: Jeder Gewichtungsfaktor (z. B. $q_1, \dots, q_4, r$) in der quadratischen Kostenfunktion muss mit physikalischer Motivation und Einheit (z. B. $1/\text{m}^2$) dokumentiert werden.
*   **Endkosten (Terminal Cost)**: Die Gewichtungen für den Endzustand $\mathbf{x}_N$ (z. B. $\mathbf{Q}_N = 10 \cdot \mathbf{Q}$) müssen separat kommentiert und motiviert werden (Sicherstellung der Stabilität am Horizontende).

---

## 5. Störgrößenvorschau (Preview / Look-Ahead)
Wenn der Controller zukünftige Störungen (z. B. ein vorausliegendes Straßenprofil $\mathbf{v}_r$) nutzt, muss erklärt werden:
*   Wie diese Information in der Realität beschafft wird (z. B. LiDAR/Kameras zur Fahrbahnabtastung, V2X-Infrastrukturdaten, Zustandsschätzer).
*   Wie sie in der Simulation implementiert ist (perfect preview Annahme als theoretisches Optimum).

---

## 6. Sprache und Struktur
*   **Deutsche Kommentare**: Wenn nicht anders spezifiziert, sind ausführliche deutsche Kommentare im Lisp-Quellcode und im generierten Python-Skript zu verwenden.
*   **Modulstruktur**: Hilfsfunktionen in Lisp zur Codegenerierung (z. B. S-Expression-Generatoren) müssen eigene Docstrings besitzen, die deren Funktionsweise erklären.

---

## 7. CasADi Schleifen-Äquivalente (For-Loop Equivalents)
*   **Vermeidung von ausgefalteten Schleifen**: Um die symbolische Graphgröße auf $O(1)$ oder $O(\log N)$ zu reduzieren, sollten für wiederholte Funktionsaufrufe CasADi-Kontrollflussäquivalente anstelle von Host-Schleifen verwendet werden.
*   **`.map(N)`**: Zu verwenden für das parallele Abbilden unabhängiger Berechnungen (z. B. Zustandskosten und Dynamik-Residuen beim Multiple Shooting).
*   **`.mapaccum(N)`**: Zu verwenden für rekursive Berechnungen (Akkumulation), bei denen das Ergebnis eines Zeitschritts als Zustandseingang für den nächsten dient (z. B. Open-Loop-Simulationen).
*   **Ausführliche Erklärungen**: Wo `.map()` oder `.mapaccum()` eingeführt werden, müssen detaillierte deutsche Kommentare den Zweck und die Funktionsweise für den Leser erläutern.

---

## 8. Abkürzungen und Fachbegriffe
*   **Begriffserklärung**: Kurze Fachbegriffe und mathematische Abkürzungen wie z. B. **QP** dürfen nicht unkommentiert verwendet werden. Schreiben Sie immer die vollständige Bedeutung in Klammern dahinter: z. B. *QP (Quadratische Programmierung / Quadratic Programming)*.

---

## 9. MPC-Horizon-Visualisierung und Vorhersagen
*   **Visualisierung der Vorschau**: Um das vorausschauende Verhalten der modellprädiktiven Regelung (MPC) zu demonstrieren, sollen zusätzlich zu den vollen Trajektorien auch die vorhergesagten Bahnen (Prediction Horizons) für charakteristische Zeitschritte geplottet werden.
*   **Typische Messpunkte**: Visualisieren Sie die geplante Trajektorie über den Horizont $N$:
    1. *Vor der Störung*: Zum Nachweis der Vorausschau (Look-Ahead).
    2. *Während der Störung*: Zum Nachweis des Reglereingriffs und von Begrenzungseffekten.
    3. *Nach der Störung*: Zum Nachweis der Stabilisierung von Nachschwingungen.
*   **Physikalische Diskussion**: Die vorhergesagten Verläufe müssen im Code direkt nach der Plot-Erzeugung in einem Kommentar-Block physikalisch interpretiert und diskutiert werden.

---

## 10. Repository-Hygiene
*   **Keine Plots einchecken**: Generierte Ergebnis-Grafiken (`.png`) dürfen nicht in Git eingecheckt werden.
*   **Lokales `.gitignore`**: Jedes Beispielverzeichnis muss ein lokales `.gitignore` enthalten, das `*.png` (und gegebenenfalls temporäre C-Code-Generierungen) ignoriert.

