# Klammer-Elision in `emit-py`: Refactoring und Korrektur

Datum: 2026-08-31
Auftrag: `plan/20260831_02_omit_paren/prompt.txt` — „code um unnuetze klammern
zu entfernen sieht sehr redundant aus“ (plus Nachtrag: die Fehler aus
`cl-cpp-generator2/plan/20260830_01_omit_paren_bug/walkthrough.md` hier
ausschliessen und testen).

## 1. Ausgangslage

`emit-py` hatte für jeden Operator eine eigene `case`-Klausel, und alle sahen
gleich aus:

```lisp
(== (let ((args (cdr code)))
      (if omit-redundant-parentheses
	  (format nil "~{~a~^==~}" (mapcar #'(lambda (x) (emit `(paren* == ,x))) args))
	  (format nil "(~{(~a)~^==~})" (mapcar #'emit args)))))
```

25 solche Klausel-Paare, dazu vier fast identische für `in`/`not-in`/`is`/`is-not`
und je zwei Varianten in `if`, `while` und `cond` für die Klammern um die
Bedingung. Die einzige Information, die sich zwischen den Klauseln unterschied,
war das Trennzeichen und gelegentlich eine Arity-Prüfung.

## 2. Refactoring: zwei Tabellen statt 25 Klauseln

Neu in `py.lisp`:

| Stelle | Zweck |
| --- | --- |
| `*infix-operators*` | Tabelle: Operator → `:separator`, `:min-args`/`:max-args`, `:style`, `:unary-format`, `:unary-legacy-format`, `:unary-operand-op` |
| `*prefix-operators*` | `not`, `~` → `:prefix` |
| `*chaining-operators*` | die Vergleichsoperatoren (Python verkettet sie) |
| `*associative-operators*` | Gruppen, die bei gleicher Präzedenz flach bleiben dürfen |
| `emit-operator` | Dispatch: liefert `NIL`, wenn der Kopf kein Operator ist, dann greift wie bisher `case` |
| `emit-infix-operator`, `emit-prefix-operator` | die eine generische Implementierung |
| `check-operator-arity` | Arity-Prüfung; erzeugt genau die alten Fehlertexte |
| `effective-operator` | Form → tatsächlich gedruckter Operator |
| `operand-needs-parentheses-p` | „muss geklammert werden?“ |
| `emit-operand` | das, was `paren*` tut |
| `emit-condition` | `if`/`elif`/`while` samt Bedingung |
| `check-operator-tables` | Ladezeit-Prüfung: jeder Operator der Tabellen steht in `*precedence*` |
| `join-strings` | Ersatz für die dynamisch gebauten `~{~a~^SEP~}`-Formatstrings |

Einen Operator hinzufügen heisst jetzt: eine Zeile in `*infix-operators*`, eine
Gruppe in `*precedence*`. Vergisst man letzteres, bricht das Laden von `py.lisp`
ab, statt stumm Klammern zu verlieren (genau der Fehler 3.2 aus dem C++-Bericht).

`emit-py` ist von 516 auf 361 Zeilen geschrumpft; `py.lisp` ist insgesamt länger
geworden, weil die Tabellen und Hilfsfunktionen dokumentiert sind.

Prüfung des reinen Refactorings: ein Dump aller 140 Testfälle plus ~120
Stress-Ausdrücke, in **beiden** Modi, war vor und nach dem Umbau byte-identisch —
bis auf die bewusst geänderten Stellen (siehe 5).

## 3. Der C++-Fehlerkatalog, auf `py.lisp` angewandt

Der Nachtrag verwies auf `cl-cpp-generator2/plan/20260830_01_omit_paren_bug`.
Die dortigen Befunde einzeln:

| C++-Befund | Status in `py.lisp` |
| --- | --- |
| 3.1 `-unary` fehlte in `*operators*` | **nicht vorhanden.** `*operators*` wird hier aus `*precedence*` berechnet, die beiden Listen können nicht auseinanderlaufen. `unary-` steht in `*precedence*`, `(- (- a b))` war schon vorher `-(a-b)`. |
| 3.2 zwei Listen, eine Wahrheit; Tippfehler `^-` | **nicht vorhanden** (siehe 3.1). Die Entscheidung hängt jetzt zusätzlich nur noch an `lookup-precedence`, und `check-operator-tables` prüft die neuen Tabellen. |
| 3.3 Abkürzung „zwei Elemente brauchen keine Klammern“ | **vorhanden und behoben.** `(** (- a) 2)` ergab `-a**2`, was Python als `-(a**2)` liest. Ersetzt durch `effective-operator`. |
| 3.4 Assoziativitätsklausel war toter Code | **vorhanden und behoben.** `(eq p0 p1)` und gleiche Zeile ⇒ gleiche Assoziativität, die Bedingung war immer `NIL`. Behoben über die Operandenposition (`:left`/`:right`). |
| 3.5 Fehler in beiden Modi (`cast`, `dot`) | **teilweise vorhanden und behoben:** `dot` und `aref` klammerten ihr Objekt nie. |
| 3.6 `paren*`-Aufrufe ohne Elternoperator | **nicht vorhanden.** |

Die konkreten Fehler, die dadurch in Python-Code entstanden sind:

| Form | vorher (falsch) | jetzt |
| --- | --- | --- |
| `(** (- a) 2)` | `-a**2` = `-(a**2)` | `(-a) ** 2` |
| `(** -2 2)` | `-2**2` = `-4` | `(-2) ** 2` = `4` |
| `(** (/ a) 2)` | `1.0/a**2` | `(1.0 / a) ** 2` |
| `(== a (== b c))` | `a==b==c` (Kettenvergleich!) | `a == (b == c)` |
| `(== (< a b) (< c d))` | `a<b==c<d` | `(a < b) == (c < d)` |
| `(in (== a b) c)` | `(a==b in c)` | `((a == b) in c)` |
| `(<< 1 (>> 8 1))` | `1<<8>>1` = 128 | `1 << (8 >> 1)` = 16 |
| `(@ a (* b c))` | `a@b*c` = `(a@b)*c` | `a @ (b * c)` |
| `(? c (? d e f) g)` | `e if d else f if c else g` | `(e if d else f) if c else g` |
| `(dot (- a b) c)` | `a-b.c` | `(a - b).c` |
| `(aref (+ a b) i)` | `a+b[i]` | `(a + b)[i]` |
| `(dot (lambda (x) x) y)` | `lambda x: x.y` | `(lambda x: x).y` |
| `(? c (ntuple a b) (ntuple d e))` | `a, b if c else d, e` | `(a, b) if c else (d, e)` |
| `(+ #C(1d0 2d0) 1)` | `type-error` beim Emittieren | `(1.0 + 1j * 2.0) + 1` |

Der Kettenvergleich ist die python-spezifische Verschärfung von 3.4: `a==b<c`
bedeutet in Python `(a==b) and (b<c)`, nicht `(a==b)<c`. Deshalb wird ein
Vergleich innerhalb eines Vergleichs **auf beiden Seiten** geklammert, nicht nur
auf der von der Assoziativität nicht bevorzugten.

### Position der Operanden

`emit-operand` bekommt jetzt `:left` (erster Operand), `:right` (jeder weitere)
oder `NIL`. Bei gleicher Präzedenz wird geklammert, wenn der Operand auf der
Seite steht, die die Assoziativität nicht bevorzugt (linksassoziativ ⇒ rechter
Operand, rechtsassoziativ ⇒ linker Operand) — es sei denn, beide Operatoren
stehen in einer Gruppe von `*associative-operators*`. Deshalb bleibt
`(+ a (+ b c))` weiterhin `a+b+c`, `(* a (@ b c))` wird aber `a*(b@c)`.

`NIL` heisst „Position egal, nur echt lockerer bindende Operanden klammern“ und
wird für das Objekt von `dot` und die Sequenz von `aref` benutzt: mit `:right`
wäre aus `(dot a (aref b i))` das ungültige `a.(b[i])` geworden.

## 4. Tests

* **`transpiler-tests.lisp`**: 20 neue Fälle mit Tag `:precedence`, einer pro
  behobener Fehlerklasse, fünf davon als `:exec-test` (Python rechnet, der Wert
  muss stimmen — `print((-2) ** 2)` &rarr; `4`).
  Gegenprobe mit dem alten `py.lisp` (`git stash`, danach
  `rm -rf ~/.cache/common-lisp`, sonst lädt SBCL das alte Fasl weiter — dieselbe
  Falle wie im C++-Bericht): 14 der 20 Fälle schlagen fehl, einer bringt den
  Testlauf mit dem `type-error` des komplexen Literals zum Absturz.
* **`paren-tests.lisp` / `run-paren-tests.sh`**: Differenztest. Der voll
  geklammerte Modus ist das Orakel; jeder Ausdruck wird zweimal emittiert, beide
  Varianten werden von `python3` per `eval` ausgewertet und die `repr`-Werte
  verglichen. Drei Schichten:
  1. 17 Unit-Tests für `effective-operator`,
  2. 21 Regressionsausdrücke (einer pro Fehler oben),
  3. 400 zufällige Ausdrücke (Tiefe 3, fester Seed, eigener LCG).

  Der Zufallstest erzeugt nur auf ganzen Zahlen totale Formen: keine Division
  durch einen Ausdruck, kein `**` mit grossem Exponenten, Shifts nur mit
  maskiertem rechten Operanden. Sonst würde der Test an einer Python-Exception
  scheitern statt an einer fehlenden Klammer.

  Gegenprobe mit dem alten `py.lisp`: 72 von 400 Zufallsausdrücken (18 %)
  weichen ab. Zur Absicherung ausserdem 8 × 1500 Ausdrücke mit Tiefe 4 und
  wechselnden Seeds — ohne Abweichung.

Beide Suiten laufen grün: 160 Transpiler-Tests, 0 Klammer-Abweichungen.

## 5. Bewusste Verhaltensänderungen

* **Fehlermeldungen bei falscher Arity** sind einheitlich: `(not a b)` meldet
  jetzt „not expects exactly one argument: (not a b)“ statt eines
  `destructuring-bind`-Fehlers. Die Texte für `/`, `//`, `%`, `**` sind
  unverändert.
* `paren*` nimmt ein optionales drittes Argument (`:left`/`:right`) und meldet
  einen Fehler, wenn es etwas anderes ist. Die Fehlermeldung bei falscher Anzahl
  heisst jetzt „paren* expects two or three arguments“.
* Im voll geklammerten Modus steht `while ( a<b ):` statt `while (a<b):` — die
  Bedingungen von `if`, `elif` und `while` gehen jetzt durch dieselbe Funktion.
* `(- a -1)` ergibt `a - (-1)` statt `a- -1` (nötig, damit `(** -2 2)` stimmt).
  In den generierten `.py`-Dateien des Repos kommt dieses Muster nicht vor, die
  Beispiele wurden deshalb nicht neu erzeugt.
* `(in (- a b) c)` ergibt `((a - b) in c)` — eine redundante, aber korrekte
  Klammer, weil der pauschale `- / // % **`-Hammer jetzt auch für die
  `:always-paren`-Operatoren greift.

## 6. Unerwartete Funde

* **Der voll geklammerte Modus liess das unäre Minus verschwinden.** `(- x)`
  wurde zu `((x))`, das Vorzeichen fehlte komplett — in *beiden*
  Codepfaden gleich falsch (Klasse 3.5 des C++-Berichts) und seit Jahren
  unbemerkt, weil der Modus nicht getestet war. Gefunden hat es der
  Differenztest, und zwar sofort: 22 der ersten 400 Zufallsausdrücke. Behoben
  über `:unary-legacy-format "(-(~a))"`.
* **Der voll geklammerte Ternär klammerte sich nicht selbst.** `(a) if (c) else (b)`
  als Objekt von `aref` ergab `([1, 2]) if (1) else ([3, 4])[1]`. Auch das fand
  der Differenztest (als Fehler im *Orakel*). Behoben; die **zweiargumentige**
  Variante bleibt bewusst unge­klammert, sie ist der Filter einer Comprehension
  (`[x for x in xs if x > 0]`) — eine Klammer würde daraus einen Generator im
  Listenliteral machen.
* **`(+ ((lambda (x) x) 1) 2)` brach mit einem `assert` ab**, weil `paren*` einen
  Symbol- oder String-Kopf erzwang. Die Asserts sind mit der Längen-Abkürzung
  verschwunden; der Aufruf eines berechneten Callables ist jetzt ein normaler
  Primärausdruck.
* **Der Hammer bleibt.** `(member parent-op '(/ // % - **))` klammert pauschal,
  sobald einer dieser Operatoren beteiligt ist. Mit der Positionsregel wäre die
  Elternseite redundant, aber sie kostet nur Lesbarkeit, und ohne sie würden sich
  die Ausgaben in ~165 Beispielen ändern.
* **`+` und `*` gelten als assoziativ**, obwohl das für Gleitkomma nur bis auf
  Rundung gilt. Das war schon vorher so und ist jetzt in
  `*associative-operators*` dokumentiert.
* **`(< a b c)` ist ein Kettenvergleich.** Die n-äre Form emittiert `a<b<c`, was
  in Python genau die CL-Semantik hat — hier passt die Sprache zufällig gut
  zusammen. Als eigener Testfall dokumentiert.
* **Zeichenketten sind ein Schlupfloch.** `(- "1/8" x)` ergibt `1/8-x`: rohe
  Strings werden unverändert eingesetzt, ohne Klammern. Das ist als
  Escape-Hatch gewollt (ein Testfall dokumentiert es), aber wer dort einen
  Operator hineinschreibt, ist selbst für die Klammern verantwortlich.
* **`(dot 5 real)`** ergibt `5.real`, was Python nicht parsen kann (`(5).real`
  wäre nötig). Zahlliterale als Objekt eines `dot` sind so selten, dass das
  hier nur notiert und nicht behandelt wird.
* Zwei Testfälle (`for-generator-basic`, `space-basic`) erzeugen Python-Fragmente,
  die `ruff format` nicht parsen kann; die Warnung im Testlauf ist alt und
  harmlos, weil Erwartung und Ergebnis identisch behandelt werden.

## 7. Reproduktion

```sh
./run-tests.sh          # 160 Transpiler-Tests
./run-paren-tests.sh    # Differenztest, 400 Zufallsausdrücke
./generate-docs.sh      # SUPPORTED_FORMS.md neu erzeugen

# härterer Zufallslauf
sbcl --noinform --disable-debugger --load paren-tests.lisp \
     --eval '(cl-py-generator/paren-tests::run-paren-tests :count 2000 :depth 4 :seed 7)' \
     --quit
```
