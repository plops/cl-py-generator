Hier ist ein strukturiertes Code-Review von **`cl-py-generator`** auf Basis der bereitgestellten Dateien (`package.lisp`, `py.lisp`, `pipe.lisp`, `cl-py-generator.asd`, `transpiler-tests.lisp`, `paren-tests.lisp`, `tests.lisp` und `SUPPORTED_FORMS.md`).

---

## 1. Gesamteindruck & Stärken

Die Codebasis ist kompakt, pragmatisch und im Bereich der Ausdrucks- und Operator-Generierung architektonisch inzwischen sehr sauber aufgestellt:

- **Tabellengetriebene Operator-Logik (`py.lisp`)**: Die Zusammenführung der früheren `case`-Zweige in `*precedence*`, `*infix-operators*`, `*prefix-operators*`, `*chaining-operators*` und `*associative-operators*` ist hervorragend. Besonders stark sind:
  - `check-operator-tables` zur Ladezeit (verhindert stille Klammerfehler durch fehlende Präzedenz-Einträge),
  - `effective-operator` (unterscheidet sauber zwischen dem Kopf der S-Expression und dem tatsächlich gedruckten Python-Operator, z. B. `(- a)` $\to$ `unary-`, `(+ a)` $\to$ Operator von `a`),
  - die korrekte Behandlung von Python-Comparison-Chaining (`*chaining-operators*`) sowie von unären/komplexen/negativen Literalen in `emit-operand`.
- **Differential-Testing (`paren-tests.lisp`)**: Die Idee, den voll geklammerten Legacy-Modus (`:omit-redundant-parentheses nil`) als Orakel gegen den gekürzten Modus über `python3` laufen zu lassen – inklusive deterministischem PRNG (`*random-state-value*`) und Regression-Suite – ist erstklassig.
- **Float-Genauigkeit (`print-sufficient-digits-f64`)**: Saubere Nutzung der ANSI-CL-Read/Print-Garantie über `*read-default-float-format*` statt fehleranfälliger Format-Strings wie `~G`.
- **Lebende Dokumentation**: `SUPPORTED_FORMS.md` direkt aus `*test-cases*` zu generieren verhindert Drift zwischen Tests und Doku.

---

## 2. Bugs & konkrete Fehlerquellen

### 2.1 Globale Mutation von `*readtable*` (`py.lisp`, Zeile 82)
```lisp
(setf (readtable-case *readtable*) :invert)
```
Das Verändern der globalen Standard-Readtable als Top-Level-Seiteneffekt einer Bibliothek ist ein klassischer Common-Lisp-Stolperstein:
- Jede nach `py.lisp` geladene Datei (inklusive Benutzer-Code, anderer ASDF-Systeme und der REPL) erbt `:invert`.
- Das Problem zeigt sich bereits in `write-notebook` (der Kommentar zu `jonathan:to-json` erklärt den Workaround) und in der Dokumentation.
- **Empfehlung**: `named-readtables` nutzen oder eine eigene Readtable lokal auf die Compile-/Load-Zeit begrenzen, statt `*readtable*` global zu überschreiben. (Hinweis: In `emit-py` wird für `dictionary :a 1` implizit erwartet, dass `:a` als `"a"` gedruckt wird – bei `:upcase` wäre es `"A"`, hier sollte man explizit `string-downcase` oder gezielte Konvertierung prüfen.)

### 2.2 Fehlende Abhängigkeit `cl-ppcre` in `cl-py-generator.asd`
In `py.lisp` (`comments`) wird `cl-ppcre:regex-replace-all` aufgerufen, aber in `cl-py-generator.asd` steht nur:
```lisp
:depends-on ("alexandria" "jonathan" "uiop" "external-program")
```
`cl-ppcre` fehlt in `:depends-on`. In `transpiler-tests.lisp` wird es nachträglich via `(ql:quickload "cl-ppcre")` geladen, wodurch der Fehler in den Tests verdeckt bleibt. In einem sauberen Image schlägt das Laden oder Ausführen von `(comments ...)` fehl.
*(Für ein einzelnes `\n`-Ersetzen ließe sich `cl-ppcre` alternativ auch komplett einsparen.)*

### 2.3 Bug im Fallback-Zweig von `paren*` (`py.lisp`)
```lisp
(paren*
  (if omit-redundant-parentheses
      ...
      ;; without the precedence machinery paren* degenerates into paren
      (format nil "(~{~a~^, ~})" (mapcar #'emit (cddr code)))))
```
`paren*` akzeptiert 2 oder 3 Argumente: `(paren* parent-op operand &optional position)`.
Wenn `position` angegeben wird, z. B. `(paren* * (+ a b) :left)`, ist `(cddr code)` gleich `((+ a b) :left)`. Im Modus `:omit-redundant-parentheses nil` wird daraus fälschlicherweise `((a+b), left)` statt `((a+b))` erzeugt. Es sollte nur `(third code)` emittiert werden.

### 2.4 Defekter `:str`-Parameter in `emit-py` (`py.lisp`)
`emit-py` deklariert `(&key code (str nil) ...)`, reicht `str` aber **ausschließlich** im `numberp`-Zweig am Ende an `(format str ...)` durch. Alle anderen Zweige verwenden `(format nil ...)` und rekursive Aufrufe geben `:str` nicht weiter. Übergibt ein Aufrufer tatsächlich einen Stream an `:str`, schreibt nur ein Zahl-Atom hinein und liefert `NIL` zurück, was umschließende `format`-Aufrufe korrumpiert.
$\to$ Parameter `:str` entweder entfernen oder konsequent über `with-output-to-string` sauber unterstützen.

### 2.5 Fehlerhafte Symbol-Syntax in `transpiler-tests.lisp` (`with-basic` / `with-as-basic`)
In `transpiler-tests.lisp` steht:
```lisp
:lisp (with (open \"f.txt\") (setf data (dot f read)))
```
Außerhalb eines Strings liest Lisp `\"f.txt\"` als **Symbol**, dessen Name wörtliche Anführungszeichen enthält! Durch `:invert` landet das dann sogar sichtbar in `SUPPORTED_FORMS.md` als:
```lisp
(with (open |"F.TXT"|) (setf data (dot f read)))
```
Es funktioniert zufällig, weil `(format nil "~a" sym)` den Symbolnamen mit den Anführungszeichen ausgibt. Gemeint ist offensichtlich `(open (string "f.txt"))` (wie bei `with-multiple`).

### 2.6 Lücken und Bugs in `parse-defun` (`py.lisp`)
1. **Rückgabetypen werden nicht durch `emit` geschickt**:
   Parameter-Typen rufen `(funcall emit type)` auf, der Rückgabetyp nutzt aber direkt `(car r)` in `~@[->~a~]`. Bei zusammengesetzten Typen wie `(values (aref list int))` oder `(values (|Optional| str))` wird rohe Lisp-Syntax statt Python ausgegeben.
2. **Keyword-Parameter ohne Default (`&key (x)`)**:
   `(format nil "= ~a" (funcall emit init))` erzeugt bei `init = nil` ungültiges Python (`x = `), während `lambda` korrekt auf `"None"` zurückfällt.
3. **Leerer Body**: `(def f ())` erzeugt `def f():\n` ohne `pass` (Python-Syntaxfehler).
4. **`&optional` und `&rest`**: Werden von `parse-ordinary-lambda-list` zwar geparst, danach aber per `(declare (ignorable ...))` stillschweigend verworfen.
5. **`captures`**: `consume-declare` sammelt `(declare (capture ...))` in `env`, `parse-defun` wertet es aber nirgends aus.

### 2.7 `return` / `return_` und `nil` (`py.lisp`)
```lisp
(return_ (format nil "return ~a" (emit (caadr code))))
(return (let ((args (cdr code)))
          (format nil "~a" (emit `(return_ ,args)))))
```
- `(return x y)` verwirft `y` stillschweigend durch `caadr` statt einen Fehler zu werfen (oder `ntuple` zu nutzen).
- `(return nil)` gibt `"return "` aus (leeres `return`) statt `return None`. Generell ist die Konvention „`nil` emittiert `""`“ fehleranfällig, sobald man Lisp-`nil` als Python-`None` meint (während `slice` `nil` explizit als „leer“ braucht).

---

## 3. Zuverlässigkeit, Sicherheit & Edge-Cases

### 3.1 `write-source` und `*file-hashes*` (`py.lisp`)
- **Hash-Tabelle mit Hashes als Keys**: `(setf (gethash (sxhash fn) *file-hashes*) (sxhash code-str))`. `sxhash` ist kein kryptographischer Hash – theoretisch sind Kollisionen möglich. Da `equal`-Hash-Tabellen direkt Strings (`namestring`) als Key und optional den Inhalt oder SHA-256 als Value speichern können, ist der Umweg über `sxhash` unnötig riskant.
- **Gelöschte Zieldatei**: Wenn die `.py`-Datei auf der Platte gelöscht oder extern geändert wird, verhindert der Cache in `*file-hashes*` das Neuschreiben, solange der Lisp-Prozess läuft. Ein zusätzliches `(probe-file fn)` würde das abfangen.
- **Cache-Update vor dem Schreiben**: Der Hash wird vor `with-open-file` gespeichert; schlägt das Schreiben fehl, bleibt der Cache inkonsistent.

### 3.2 `write-notebook` (`py.lisp`)
- **Feste Temp-Datei ohne Cleanup**:
  `notebook-cell-source` schreibt jede Python-Expression über `write-source` in eine feste Datei `<tmpdir>/cl-py-generator-cell.py`, löscht sie danach nicht und ist bei paralleler Ausführung nicht race-condition-sicher.
- **Kombination mit `*file-hashes*`**: Wenn eine Zelle identischen Code wie eine vorherige hat, überspringt `write-source` das Schreiben (funktioniert nur zufällig, solange niemand die Temp-Datei löscht/verändert).
- **Performance**: Für jede einzelne Expression in jeder Zelle wird `ruff format` als externer Prozess gestartet. Sinnvoller wäre `emit-py` im Speicher + optional einmaliges Formatieren.

### 3.3 `pipe.lisp`
- Verwendet nicht-exportierte SBCL-Interna (`sb-impl::process-%status`, `sb-impl::process-pty`) statt der offiziellen `sb-ext:process-status` und `sb-ext:process-pty`.
- `run` ist nicht in `package.lisp` exportiert und prüft via `assert`, ohne Thread-Synchronisation zum Reader-Thread.

### 3.4 Operator-Arity & Raw-Strings (`py.lisp`)
- **Fehlende `:min-args`-Prüfungen**: Für `+`, `-`, `*`, `@`, `<`, `>`, `<=`, `>=`, `==`, `!=`, `&`, `^`, `|`, `and`, `or`, `<<`, `>>` ist `:min-args` `nil`. Dadurch emittieren z. B. `(-)` $\to$ `""` und `(== a)` $\to$ `"a"` ohne Fehler. (Bei `and`/`or`/`+`/`*` kann 1-stellig für Makro-Expansionen gewollt sein – bei Vergleichsoperatoren oder 0 Argumenten ist es fast sicher ein Bug.)
- **Raw-Strings in Ausdrücken**: In `emit-operand` wird `(stringp arg)` unbesehen zurückgegeben. `(- "1/8" x)` funktioniert, aber `(* "1+2" 3)` wird zu `1+2*3` (Fehl-Gruppierung). Das sollte zumindest dokumentiert sein.
- **Kein Escaping in `(string ...)`**: `(string "a\"b")` erzeugt ungültiges Python `"a"b"`.

---

## 4. Tests & Wartbarkeit

- **Ruff-Normalisierung in `transpiler-tests.lisp`**:
  - Da sowohl `actual` als auch `expected` durch `ruff format` laufen, werden Whitespace-/Leerzeilen-Fehler in `emit-py` (wie z. B. fehlende Leerzeichen um `=` oder `->`) durch Ruff überdeckt.
  - Tests für Einzelfragmente wie `for-generator` (`i*2 for i in range(3)`) oder `space` (`alpha beta`) sind für sich genommen kein gültiges Python; `ruff format` schlägt dort mit einer Warnung fehl und fällt auf den Roh-String-Vergleich zurück.
  - Pro Testfall zwei Subprozesse (`ruff`) zu starten, bremst die Suite unnötig aus.
- **Erweiterungspotenzial in `paren-tests.lisp`**:
  `*random-forms*` prüft bisher keine `in`/`not-in`/`is`/`is-not`, `@`, `/`, `aref`, `dot`, `lambda` oder `ntuple`. Gerade `in` vs. `<` (Comparison-Chaining) und `dot`/`aref` auf komplexen Teilausdrücken wären wertvolle Ergänzungen im Fuzzer.
- **Package-Exports (`package.lisp`)**:
  - `not` fehlt im `:export` (funktioniert nur durch `(:use :cl)`), während `and`, `or`, `if`, `cond`, `lambda` etc. exportiert sind.
  - In `py.lisp` und `transpiler-tests.lisp` liegt noch viel auskommentierter Altcode (`#+nil`, `:optima`-Reste), der entfernt werden kann.

---

## 5. Priorisierte Handlungsempfehlungen

1. **`cl-ppcre` in `cl-py-generator.asd` ergänzen** (oder in `comments` ohne Regex ersetzen).
2. **Bug in `paren*`-Fallback fixen**: `(third code)` statt `(cddr code)` emittieren.
3. **`with-basic` / `with-as-basic` in `transpiler-tests.lisp` korrigieren**: `(string "f.txt")` statt `\"f.txt\"` und `SUPPORTED_FORMS.md` neu generieren.
4. **Globales `(setf (readtable-case *readtable*) :invert)` entschärfen**: Auf eine lokale/benannte Readtable umstellen, um Aufrufer-Code nicht zu beeinflussen.
5. **`parse-defun` härten**: `(funcall emit (car r))` für den Rückgabetyp verwenden, leere Bodies mit `pass` abfangen, `&key` ohne Init-Wert auf `None` setzen.
6. **`write-notebook` & `write-source` bereinigen**: Echte temporäre Dateien über `uiop:with-temporary-file` in `write-notebook`, `probe-file` + `equal`-Hash-Table in `write-source`, und unbenutztes `:str` in `emit-py` entfernen.
7. **In `pipe.lisp` auf `sb-ext:process-status` / `sb-ext:process-pty` wechseln**.


Nachfolgend findest du für jedes im Review gefundene Problem konkrete Lösungsansätze. Wo es mehrere sinnvolle Wege gibt, sind die Vor- und Nachteile diskutiert und **vom besten zum schwächsten Ansatz sortiert**.

---

## 1. Globale Mutation von `*readtable*` (`py.lisp`)
**Problem:** `(setf (readtable-case *readtable*) :invert)` verändert beim Laden der Bibliothek global die Readtable für die gesamte Lisp-Session und allen nachfolgend geladenen Code.

1. **Beste Lösung: Lokale `named-readtables` + explizite Groß-/Kleinschreibung im Emitter**
   - Eine eigene Readtable (z. B. `(defreadtable :cl-py-generator (:merge :standard) (:case :invert))`) wird nur in den Bibliotheksdateien aktiviert, die sie wirklich benötigen (`(in-readtable :cl-py-generator)`).
   - Zusätzlich im Emitter bei Keyword-Argumenten (`dictionary`, Funktionsaufrufe mit `:bar`) explizit `string-downcase` verwenden (sofern der Name nur aus Großbuchstaben besteht), damit Benutzer die Bibliothek auch mit Standard-`:upcase`-Readtable problemlos nutzen können.
   - *Vorteil:* Saubere Kapselung, kein „Verschmutzen“ fremder Systeme, kompatibel mit dem restlichen Common-Lisp-Ökosystem.
   - *Nachteil / Abwägung:* Bestehende `example/*/gen*.lisp`-Skripte, die sich implizit auf das globale `:invert` verlassen, müssten ggf. ein `(in-readtable :cl-py-generator)` an den Dateianfang bekommen.

2. **Alternative (Minimal-invasiv, aber weiterhin global für Beispiele): Readtable beim Laden sichern/wiederherstellen**
   - Nur die Dateien von `cl-py-generator` mit gebundenem `*readtable*` laden und für die alten Beispiele ein separates Hilfsmakro/Setup bereitstellen.
   - *Nachteil:* Lösen muss man die Case-Frage für Benutzer ohnehin, daher ist `named-readtables` sauberer.

---

## 2. Fehlende `cl-ppcre`-Abhängigkeit (`cl-py-generator.asd` / `py.lisp`)
**Problem:** `comments` in `py.lisp` ruft `cl-ppcre:regex-replace-all` auf, obwohl `cl-ppcre` nicht in `:depends-on` der `.asd`-Datei steht.

1. **Beste Lösung: `cl-ppcre` in `py.lisp` durch Bordmittel ersetzen**
   - Das Ersetzen von `#\Newline` durch `~%# ` in `comments` lässt sich in 5 Zeilen Standard-CL (oder mit `uiop:split-string`) lösen:
     ```lisp
     (with-output-to-string (s)
       (loop for ch across arg do
         (if (char= ch #\Newline)
             (write-string (load-time-value (format nil "~%# ")) s)
             (write-char ch s))))
     ```
   - *Vorteil:* Eine externe Laufzeitabhängigkeit weniger; `cl-py-generator.asd` bleibt schlank. (In `transpiler-tests.lisp` lässt sich `\\r\\n?` ebenfalls leicht durch `uiop` oder einfaches Zeichen-Filtern ersetzen.)

2. **Alternative: `"cl-ppcre"` in `cl-py-generator.asd` unter `:depends-on` eintragen**
   - *Vorteil:* Einzeilige Änderung, sofortige Korrektur.
   - *Nachteil:* Schleppt eine komplette Regex-Engine für eine einzige triviale Zeichenersetzung mit.

---

## 3. Bug im Fallback-Zweig von `paren*` (`py.lisp`)
**Problem:** Bei `:omit-redundant-parentheses nil` und 3 Argumenten wie `(paren* * (+ a b) :left)` gibt `(mapcar #'emit (cddr code))` das Positions-Keyword `:left` mit aus: `((a+b), left)`.

1. **Beste Lösung: Validierung vor der Verzweigung, im Fallback nur `(third code)` klammern**
   - Die Längen- und Positionsprüfung (`'(3 4)`, `'(nil :left :right)`) **vor** dem `(if omit-redundant-parentheses ...)` ausführen.
   - Im Legacy-Zweig (`nil`) exakt ein geklammertes Element zurückgeben:
     ```lisp
     (format nil "(~a)" (emit (third code)))
     ```
   - *Vorteil:* Verhindert fehlerhafte Tupel-Ausgaben und fängt ungültige `paren*`-Aufrufe in beiden Modi ab.

2. **Alternative: Im Fallback-Zweig `(subseq code 2 3)` bzw. `(third code)` ohne Vorab-Check verwenden**
   - *Nachteil:* Ungültige Argumente werden im Orakel-Modus nicht gemeldet.

---

## 4. Defekter `:str`-Parameter in `emit-py` (`py.lisp`)
**Problem:** `:str` wird nur im `numberp`-Zweig an `format` weitergereicht; alle anderen Zweige geben einen String zurück und ignorieren `:str`.

1. **Beste Lösung: `:str` aus der Signatur von `emit-py` entfernen**
   - Prüfen, ob irgendwo in `example/` `:str` übergeben wird (falls nein: komplett streichen). `emit-py` liefert dann konsistent immer einen `string` zurück.
   - *Vorteil:* Klare, vorhersehbare API ohne toten Code.

2. **Alternative (100 % abwärtskompatibel): Zentral am Ende von `emit-py` auf `str` schreiben**
   - Alle inneren Konstrukte geben wie bisher Strings zurück (rekursive Aufrufe mit `:str nil`). Ganz außen am Ende von `emit-py`:
     ```lisp
     (if str
         (progn (write-string result str) nil)
         result)
     ```
   - *Vorteil:* Macht `:str` für **alle** S-Expressions funktionsfähig (nützlich z. B. für direktes Streamen in Dateien), ohne bestehende Aufrufer zu brechen.

---

## 5. Fehlerhafte Symbol-Syntax `\"f.txt\"` in `transpiler-tests.lisp`
**Problem:** `:lisp (with (open \"f.txt\") ...)` wird als Symbol `|"F.TXT"|` statt als String gelesen (und erscheint so auch in `SUPPORTED_FORMS.md`).

1. **Beste Lösung: Testfälle korrigieren und Doku neu generieren**
   - In `with-basic` und `with-as-basic` `(open (string "f.txt"))` verwenden (analog zu `with-multiple`) und anschließend `./generate-docs.sh` ausführen.
   - *Vorteil:* Korrekte DSL-Nutzung und saubere `SUPPORTED_FORMS.md`.

---

## 6. Lücken und Bugs in `parse-defun` (`py.lisp`)
**Problem:** Rückgabetypen werden nicht durch `emit` geschickt; `&key` ohne Default erzeugt `x = `; leere Bodies ergeben ungültiges Python; `&optional`/`&rest` werden ignoriert; `captures` ist ungenutzt.

1. **Beste Lösung: Vollständige Härtung von `parse-defun`**
   - **Rückgabetyp**: `(funcall emit (car r))` statt rohem `(car r)` verwenden (ermöglicht z. B. `(values (aref list int))` oder `(values (|Optional| str))`) und sauber mit `" -> ~a"` formatieren.
   - **Keyword-Defaults**: Bei `init = nil` explizit `"= None"` statt `"= "` ausgeben (oder `if init ... else "= None"`, analog zu `lambda`).
   - **Leerer Body**: `(do ,@(or body '(pass)))`, damit `(def f ())` gültiges Python mit `pass` erzeugt.
   - **`&optional` / `&rest`**: Entweder sauber unterstützen (`&optional` als `p=default` bzw. `p=None`, `&rest` als `*args`) oder – falls bewusst nicht unterstützt – mit einem klaren `(error ...)` abbrechen statt Parameter stillschweigend zu verschlucken.
   - **`captures`**: Prüfen, ob Überbleibsel aus einem C++-Generator; falls ungenutzt, aus `consume-declare` entfernen (oder explizit als `nonlocal` emittieren, falls das gemeint war).

2. **Alternative (Minimale Sofortkorrektur)**
   - Nur `(funcall emit (car r))`, `"= None"` bei leerem Default und `pass` bei leerem Body beheben sowie für `&optional`/`&rest` einen `error` werfen.
   - *Vorteil:* Minimaler Eingriff ohne neue Semantik bei Lambda-Listen.

---

## 7. `return` / `return_` und `nil` (`py.lisp`)
**Problem:** `(return x y)` verwirft `y` stillschweigend durch `caadr`; `(return nil)` erzeugt `"return "` statt `"return None"`.

1. **Beste Lösung: `return` direkt implementieren + Arity-Check**
   - `return_` kann als einfacher Wrapper bleiben oder entfallen, und `return` unterscheidet sauber:
     - `0` Argumente `(return)` $\to$ `"return"`
     - `1` Argument `(return expr)` $\to$ `(format nil "return ~a" (if (null expr) "None" (emit expr)))`
     - `>1` Argumente $\to$ `(error "return expects at most 1 argument; use (return (ntuple ...)) for multiple values")`
   - *Abwägung:* Falls in alten `example/`-Dateien `(return nil)` als „leeres `return`“ genutzt wurde, würde `"return None"` das Verhalten dort leicht ändern (allerdings semantisch ins Bessere, da Lisp-`nil` eher `None` entspricht; falls Rückwärtskompatibilität Vorrang hat, zunächst nur den `>1`-Arity-Error einführen).

2. **Alternative: Mehrere Argumente zu `ntuple` zusammenfassen**
   - `(return x y)` würde dann `return x, y` erzeugen.
   - *Nachteil:* Macht die DSL mehrdeutig neben `(return (ntuple x y))`; ein expliziter Fehler ist sauberer.

---

## 8. `write-source` und `*file-hashes*` (`py.lisp`)
**Problem:** `sxhash` als Key & Value (Kollisionsrisiko), Hash wird vor dem Schreiben gesetzt, gelöschte Dateien werden wegen Cache-Treffer nicht neu erstellt.

1. **Beste Lösung: `equal`-Hash-Table mit Dateipfad + `probe-file`-Prüfung + Update nach Schreiben**
   - `*file-hashes*` mit `(make-hash-table :test 'equal)` initialisieren.
   - Als Key `(namestring fn)` verwenden, als Value `code-str` (oder einen echten Digest wie SHA-256, wobei Python-Dateien i. d. R. wenige KB groß sind und der direkte String-Vergleich `string=` kollisionsfrei ist).
   - Nur überspringen, wenn `(and exists (probe-file fn) (string= old-code code-str))`.
   - `(setf (gethash key *file-hashes*) code-str)` erst **nach** dem erfolgreichen `with-open-file` ausführen.
   - *Vorteil:* Keine Hash-Kollisionen, fehlende Dateien werden neu erzeugt, konsistenter Zustand bei I/O-Fehlern. (Der Cache bleibt weiterhin sinnvoll, weil auf Platte die von `ruff` formatierte Fassung liegt, während hier gegen den rohen `emit-py`-Output geprüft wird.)

2. **Alternative: Verzicht auf `*file-hashes*`, stattdessen nur schreiben, wenn unformatiert != Dateiinhalt**
   - *Nachteil:* Da `format-python-file` die Datei auf Platte nachträglich umformatiert, stimmt der Dateiinhalt nicht mehr mit `code-str` überein – man müsste jedes Mal zuerst `ruff` laufen lassen, um vergleichen zu können. Daher ist Lösung 1 besser.

---

## 9. `write-notebook` (`py.lisp`)
**Problem:** Feste Temp-Datei `"cl-py-generator-cell"` (Race Condition, kein Cleanup, Interaktion mit `*file-hashes*`), und für jede Expression wird ein eigener `ruff`-Prozess gestartet.

1. **Beste Lösung: Direkt `emit-py` im Speicher + einmalige temporäre Formatierung pro Zelle**
   - In `notebook-cell-source` für jede Form in `rest` `(emit-py :clear-env nil :code p)` aufrufen, die Strings einer Zelle mit Newlines verbinden und – falls ein Formatter aktiv ist – einmalig über `uiop:with-temporary-file (:type "py" :keep nil)` mit `format-python-file` formatieren und danach in Zeilen für das Notebook-JSON aufspalten.
   - *Vorteil:* Keine kollidierenden Dateien, kein Müll in `/tmp`, keine `*file-hashes*`-Seiteneffekte und deutlich schneller bei größeren Notebooks.

2. **Alternative: `uiop:with-temporary-file` mit direktem `emit-py` pro Expression**
   - *Nachteil:* Immer noch ein Subprozess pro Expression statt pro Zelle.

---

## 10. SBCL-Interna in `pipe.lisp`
**Problem:** Nutzung nicht-exportierter Interna `sb-impl::process-%status` und `sb-impl::process-pty`.

1. **Beste Lösung: Dokumentierte `sb-ext`-API verwenden**
   - `sb-impl::process-%status` durch `(sb-ext:process-status *python*)` und `sb-impl::process-pty` durch `(sb-ext:process-pty *python*)` ersetzen.
   - `read-line` mit EOF-Wert `nil` statt `'foo` betreiben (`loop for line = (read-line pty nil nil) while line ...`).
   - *Vorteil:* Stabil über SBCL-Versionen hinweg bei vollem Erhalt der `:pty t`-Funktionalität.

2. **Alternative: Portierung auf `uiop:launch-program`**
   - *Vorteil:* Funktioniert auch auf CCL, ECL, ABCL usw.
   - *Nachteil:* `uiop:launch-program` bietet standardmäßig kein PTY (`:pty t`), was bei interaktiver Python-REPL-Nutzung (Buffering/Prompt) zu Problemen führen kann.

---

## 11. Operator-Arity, Raw-Strings & String-Escaping (`py.lisp`)
### 11a. Fehlende `:min-args` in `*infix-operators*`
1. **Beste Lösung: Differenzierte `:min-args`-Regeln**
   - Für `+`, `*`, `and`, `or`, `logand`, `&`, `logxor`, `^`, `logior`, `|` mindestens `:min-args 1` setzen (so funktionieren variadische Makros à la `(and ,@conds)` weiterhin, aber `(+)`, `(-)` mit 0 Args werden abgefangen).
   - Für Vergleichsoperatoren (`<`, `>`, `<=`, `>=`, `==`, `!=`), `<<`, `>>`, `@` explizit `:min-args 2` setzen, da 1 Operand hier in Python keinen Sinn ergibt.
   - Für `-` `:min-args 1` ergänzen (analog zu `/`), damit `(-)` einen klaren Fehler wirft.

2. **Alternative: Strikte `:min-args 2` für alle binären Operatoren (außer `-` und `/`)**
   - *Nachteil:* Bricht Makros, die eventuell `(and x)` oder `(+ x)` generieren (und ein Test in `paren-tests.lisp` prüft aktuell explizit `(or 255)` und `(+ (* a b))`!). Daher ist Lösung 1 deutlich besser.

### 11b. Raw-Strings in Ausdrücken & fehlendes Escaping in `(string ...)`
1. **Beste Lösung für Raw-Strings: Beibehalten + dokumentieren (und AST-Formen bevorzugen)**
   - Bare Strings sind bewusst „rohes Python“ (wie `":"` in `aref` oder `"1/8"` in `(- "1/8" x)`). Ein automatisches Klammern in `emit-operand` würde Tricks wie `":"` zerstören. Stattdessen die Warnung in `SUPPORTED_FORMS.md`/Doku festhalten, dass zusammengesetzte Raw-Strings als Operanden manuell geklammert oder als S-Expression formuliert werden sollten.
2. **Beste Lösung für String-Escaping: Neue Form (z. B. `pystring` oder `:escape t`) oder vorsichtige Prüfung**
   - Automatisches Escaping in `(string ...)` könnte bestehende Beispiele brechen, die bereits manuell `\"` escapen. Sicherer ist eine neue Form (z. B. `str` / `text`) oder eine Prüfung aller `example/`-Dateien vor der Umstellung von `string`.

---

## 12. Test-Suite, Fuzzer & Altcode (`transpiler-tests.lisp`, `paren-tests.lisp`, `package.lisp`)
1. **Test-Runner (`transpiler-tests.lisp`)**:
   - **Beste Lösung**: Testfälle, die kein vollständiges Python sind (`for-generator`, `space`), mit `:ruff nil` markieren, damit `ruff format` keine irreführenden Warnungen auswirft. Außerdem optionalen Roh-Check (ohne `ruff`) oder Batch-Formatierung aller Testdateien auf einmal ergänzen, um 200 Subprozess-Starts zu sparen und echte Formatierungsfehler von `emit-py` sichtbar zu machen.
2. **Fuzzer (`paren-tests.lisp`)**:
   - **Beste Lösung**: `*random-forms*` um `in`/`not-in` (mit fester kleiner Liste), `/`, `aref` (auf einer Listen-Expression), `dot` (z. B. `bit_length`), `lambda`-Aufrufe und 2-arg `?` erweitern, um auch diese Grenzbereiche automatisch per Differential-Test abzusichern.
3. **Exports & Altlasten (`package.lisp`, `py.lisp`)**:
   - **Beste Lösung**: `:not` in `package.lisp` exportieren; auskommentierte `#+nil`-Leichen und `optima`-Kommentare entfernen (über `git` bleiben sie jederzeit auffindbar); `tests.lisp` so umstellen, dass beim bloßen Laden keine Datei nach `/dev/shm/o` geschrieben wird (oder in eine Funktion kapseln).
   
   
