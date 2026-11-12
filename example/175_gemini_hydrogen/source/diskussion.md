Für eine erste qualitative Einordnung sind die Ergebnisse schon sehr aussagekräftig – man kann damit bereits den **Bias-Varianz-Tradeoff** und den **Unterschied zwischen SavGol und Gauß-Filter** sauber erklären.

Für eine wirklich **wasserdichte Diskussion** fallen im Plot aber drei Dinge auf, die noch nach einer kleinen Nachschärfung schreien:

---

### 1. Was sich aus deinen aktuellen Zahlen bereits brillant herauslesen lässt

Deine Zahlen erlauben direkt eine exakte **Zerlegung in Bias und Varianz** ($\text{RMSE}^2 \approx \text{Bias}^2 + \text{Rauschen}^2$ mit $\sigma_{\text{out}} = \text{NoiseGain} \cdot 0.05$):

* **`np.gradient`:** Der gemessene Noise-Gain ist exakt das theoretische Limit $\frac{1}{\sqrt{2}\,\Delta x} = \frac{1}{\sqrt{2}\cdot 0.02} = 35.355$. Damit ist das erwartete Rauschen $35.355 \cdot 0.05 = 1.768$, was fast 1:1 dem gemessenen RMSE (`1.733`) entspricht.
* **Der eigentliche Clou (SavGol vs. Gauß):**
  Obwohl der Gauß-Filter das weiße Rauschen sogar etwas **stärker dämpft** (`NG = 0.434` vs. `0.453`), hat er den **schlechteren RMSE** (`0.02703` vs. `0.02304`). Warum?
  * **SavGol (`w=97, p=3`):** Rauschanteil $= 0.453 \cdot 0.05 = 0.02265 \implies \text{Bias} = \sqrt{0.02304^2 - 0.02265^2} \approx \mathbf{0.0042}$. SavGol arbeitet hier fast am reinen Rauschlimit.
  * **Gauss (`s=0.247`):** Rauschanteil $= 0.434 \cdot 0.05 = 0.02170 \implies \text{Bias} = \sqrt{0.02703^2 - 0.02170^2} \approx \mathbf{0.0161}$ – also **fast 4× mehr systematischer Fehler**.
  * **Physikalische Erklärung im Frequenzgang:** Ein Gauß-Filter flacht $|H(f)|$ schon nahe $f = 0$ quadratisch ab, während `SavGol (p=3)` dem idealen Differentiator $2\pi f$ im Durchlassbereich deutlich länger flach folgt und erst knapp vor $f_{-3\text{dB}} = 0.65\,\text{Hz}$ steil abknickt (genau dort, wo das Spektrum des $\sigma=1$-Gaußsignals ohnehin abklingt).

---

### 2. Wo der aktuelle Plot noch Schwächen hat (warum man nachlegen sollte)

1. **Das Optimum `w=97` klebt direkt an der Suchgrenze (`101`)!**
   Unten links sieht man, dass `poly=3` (orange) bei `97` und `poly=5` (grün) sogar bei `101` noch kaum wieder ansteigen. Der Suchbereich `5..101` (entspricht nur `2.0` $x$-Einheiten bei einer Signalbreite von $6\sigma = 6.0$) war schlicht **zu schmal**, um den Anstieg durch Bias bei großen Fenstern klar zu zeigen. Zudem drückt die lineare Y-Achse die Minima platt.
   * **Fix:** Fenster bis `w=251` laufen lassen und `ax10.set_yscale("log")` verwenden – dann sieht man für alle drei Ordnungen die klassische **U-Kurve** (Varianz links, Bias rechts) gestochen scharf.

2. **Im Frequenzgang überstrahlt `Ideal: 2*pi*f` (bis `38`) den eigentlichen Vergleich:**
   Unten rechts erkennt man kaum, dass **SavGol im Sperrbereich Nebenkeulen (Sidelobes)** bis `~0.6` hat, während **Gauss** oberhalb von `~1.8 Hz` komplett auf Null abfällt.
   * **Fix:** Auf `f in [0, 3.0]` und `y in [0, 5]` zoomen (oder `semilogy` für $|H(f)|$) – dann wird der Tradeoff *„steile Flanke + Nebenkeulen (SavGol) vs. runde Flanke + perfekte Sperrdämpfung (Gauß)“* sofort sichtbar.

3. **Nur eine einzige Zufalls-Realisierung (`seed=42`):**
   Weil das RMSE-Tal bei `w = 80..110` extrem flach ist, kann das exakte `best_win` bei einem anderen Seed leicht zwischen z. B. `85` und `105` springen.

---

### 3. Sinnvolle Zusatz-Simulationen (falls du noch mehr herausholen willst)

Je nachdem, wie tief die Diskussion gehen soll, lohnen sich vor allem diese drei Ergänzungen:

1. **Monte-Carlo-Mittelung (z. B. 50–100 Seeds):** Zeigt Mittelwert $\pm$ Standardabweichung des RMSE über `window_length` und bestätigt, dass `p=3` systematisch und nicht nur zufällig besser ist als `p=1`, `p=5` und `Gauss`.
2. **Sweep über das SNR / Rauschlevel ($\sigma_{\text{noise}} \in [0.005, 0.2]$):** Zeigt schön, wie sich $w_{\text{opt}}$ mit steigendem Rauschen zu breiteren Fenstern verschiebt (und wann `p=1` oder `Gauss` ggf. aufholen).
3. **Stresstest mit schmalbandigem Störer (z. B. $0.05 \cdot \sin(2\pi \cdot 2.0\, x)$ statt nur weißem Rauschen):** Ein echtes Highlight für die Diskussion – weil SavGol bei $2\,\text{Hz}$ noch Nebenkeulen hat, schlägt dort plötzlich der **Gauß-Filter** den SavGol-Filter deutlich!

**Fazit:** Mit `windows` bis `251` + `log`-Y-Achse im Bias-Varianz-Plot und einem Zoom im Frequenzgang ist der eine Durchlauf bereits komplett rund für eine saubere Diskussion; für wissenschaftliche Belastbarkeit wäre noch die Monte-Carlo-Mittelung über mehrere Seeds das wichtigste Upgrade.
