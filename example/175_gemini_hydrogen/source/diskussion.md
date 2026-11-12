Jetzt ist die Simulation **komplett rund und absolut belastbar** für eine fundierte Diskussion. Im neuen Plot greifen Zeitbereich, Optimierungskurve und Frequenzraum wie Zahnräder ineinander:

---

### 1. Bias-Varianz-Optimierung (unten links)
* **Vorher vs. Nachher:** Durch die Erweiterung bis `w = 251` sieht man jetzt das echte globale Optimum für `poly=3` bei **`w = 125`** (statt vorher am Suchrand bei `97`), mit einem Monte-Carlo-RMSE von **`0.02023`** gegenüber **`0.02671`** beim optimierten Gauß-Filter (~24 % besser).
* **Alle drei U-Kurven sichtbar:**
  * **`poly=1`** erreicht sein Minimum bei `w ≈ 55` und berührt dort exakt die gestrichelte **`Gauss Opt.`-Linie (`0.0267`)**. Das ist kein Zufall: Sowohl `SavGol (p=1)` (identisch mit `p=2`) als auch der Gauß-Filter haben eine Fehlerordnung von $O(h^2)$ – keiner hat beim Bias-Verhalten einen asymptotischen Vorteil über den anderen.
  * **`poly=3`** (orange) unterbietet den Gauß-Filter im breiten Bereich `w ≈ 80..175` deutlich (auch inklusive der `±1σ`-Fehlerbänder über 40 Realisierungen – der Vorteil ist also statistisch signifikant, kein Ausreißer eines einzelnen Seeds).
  * **`poly=5`** (grün) verschiebt das Optimum noch weiter nach rechts (`w ≈ 195..200`), bleibt aber knapp über `poly=3`, weil das 5.-Grad-Polynom mehr Rauschvarianz einfängt.
* **Optimales Bias-Varianz-Gleichgewicht:** Bei `w = 125` sind Bias (`0.01300`) und Rauschanteil (`0.01550`) nahezu ausbalanciert – genau das charakteristische Zeichen eines gut getroffenen Bias-Varianz-Optimums.

---

### 2. Warum SavGol beim weißen Rauschen gewinnt – aber bei `2.0 Hz` verliert (unten rechts)
Der Frequenzgang-Zoom erklärt **beide** Ausgaben gleichzeitig:

1. **Weißes Rauschen (`SavGol: 0.0202` vs. `Gauss: 0.0267`):**
   * Das graue **Signalspektrum $|g'(f)|$** liegt fast vollständig unterhalb von **`0.5 Hz`**.
   * `SavGol (w=125, p=3)` bleibt bis ~`0.35 Hz` sehr nah an der idealen Gerade $2\pi f$ (kleiner Bias) und fällt dann bei **`f_-3dB = 0.50 Hz`** steil ab.
   * Der **Gauß-Filter** dämpft im Signalband etwas früher (`Bias: 0.01559` vs. `0.01300`), lässt aber zwischen **`0.55 Hz` und `1.6 Hz`** (grün über orange!) unnötig viel weißes Rauschen passieren (`Noise: 0.02169` vs. `0.01550`). SavGol schlägt den Gauß-Filter also **gleichzeitig bei Bias und Rauschunterdrückung**.

2. **Schmalband-Störtest bei `2.0 Hz` (`SavGol: 0.02610` vs. `Gauss: 0.01967`):**
   * Bei **`f = 2.0 Hz`** ist der Gauß-Filter praktisch auf **Null** abgeklungen, während `SavGol` dort eine seiner **Nebenkeulen (`|H(2.0)| ≈ 0.3–0.4`)** hat.
   * Dadurch kehrt sich das Ergebnis beim schmalbandigen Störer um: Der **Gauß-Filter gewinnt um ~25 %**, weil er ohne Überschwinger/Nebenkeulen sperrt.

---

### Fazit für deine Diskussion
Du brauchst **keine weiteren Simulationen** mehr – mit diesem einen 2×2-Plot und den drei Kennzahlen-Zeilen kannst du die komplette Geschichte sauber belegen:
1. Naive finite Differenzen (`np.gradient`) sind wegen $\text{NoiseGain} = 35.4$ bei verrauschten Ableitungen unbrauchbar (`RMSE = 1.785`).
2. `SavGol (p=3)` verhält sich wie ein optimierter Tiefpass-Differentiator mit flachem Durchlassbereich und steiler Flanke – ideal bei **glattem Signal + weißem Rauschen**.
3. `Gauss-Deriv` hat das schlechtere Durchlass-/Flankenverhalten, aber ein **monoton abfallendes Sperrband ohne Nebenkeulen** – ideal bei **hochfrequenten schmalbandigen Störern** oder wenn keinerlei Ringing toleriert wird.
