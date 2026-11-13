# Abnahme-Check: Wie gut hat Muse Spark das Hydrogen-Review umgesetzt?

Stell dir vor, du übergibst einem Kollegen ein 12-Punkte-Review voller subtiler Lisp-Fallstricke – von mutierten globalen Readtables über vergessene ASDF-Abhängigkeiten bis zu stillen Fehlern im Datei-Cache – und bekommst ein fertiges Refactoring samt Test-Suite und Walkthrough zurück.

Die kurze Antwort vorweg: **Muse Spark hat richtig starke Arbeit geleistet.** Alle 12 Review-Punkte wurden nicht nur oberflächlich „weggefixt“, sondern inhaltlich durchdrungen. Besonders erfreulich: Während der Arbeit an den Tests hat die KI zwei versteckte Probleme selbst entdeckt, die im ursprünglichen Review noch gar nicht standen – darunter einen echten Bug im *Orakel-Modus* des Differential-Testers.

Trotzdem lohnt sich beim Code-Check bekanntlich der zweite Blick unter die Motorhaube. Und tatsächlich: Zwei kleine Inkonsistenzen haben sich doch noch versteckt. Gehen wir gemeinsam Schritt für Schritt durch das Ergebnis.

---

## 1. Die Reise durch die Fixes: Was besonders gut gelungen ist

### A. Die „unsichtbare“ Readtable-Falle sauber gelöst
Das kniffligste Problem aus dem Review war der globale Eingriff in `*readtable*` (`:invert`). Ein naives Entfernen dieser einen Zeile hätte sofort die halbe Test-Suite und alte Nutzer-Skripte zerrissen.

Muse Spark hat genau verstanden, warum die alte Suite überhaupt grün war:
> SBCLs Drucker kompensiert `:invert` beim Ausgeben automatisch – er war es, der intern großgeschriebene Symbole (`DATA`) stillschweigend wieder klein (`data`) druckte.

Die gewählte Lösung ist elegant zweigleisig:
1. **Keine globale Mutation mehr**: `py.lisp` rührt `*readtable*` nicht mehr an. Wer aus fremden Paketen (`cl-user`) mit Standard-`:upcase`-Readtable Code generiert, bleibt verschont.
2. **Explizites Downcasing über `py-symbol-name`**: Alle Stellen, die Python-Bezeichner ausgeben (`emit-py`, `parse-defun`, `class`, `import`, `try`-Zweige, Funktionsaufrufe und Keyword-Args), wandern jetzt durch `py-symbol-name`. Namen ohne Kleinbuchstaben (`FOO`, `:A`) werden klein gedruckt (`foo`, `a`); Namen mit gemischter Schreibung (`|MyVar|`) bleiben erhalten.
3. **Lokale Readtable für Tests (`named-readtables`)**: `package.lisp` definiert eine benannte Readtable `:cl-py-generator`, und `transpiler-tests.lisp` sowie `paren-tests.lisp` aktivieren sie *nur für ihre eigene Datei*. Sogar an das `let ((*readtable* ...))` beim Drucken in `generate-documentation` wurde gedacht – sonst hätte `SUPPORTED_FORMS.md` plötzlich Großbuchstaben gezeigt.

### B. Der Fuzzer findet einen Bug im eigenen Orakel
Beim Erweitern von `paren-tests.lisp` um `dot`, `aref`, `in`, `not-in`, `/` und `lambda`-Aufrufe stolperte Muse Spark über ein klassisches Test-Paradoxon: **Das Orakel selbst lag falsch.**

`(dot -3 (bit_length))` wurde im voll geklammerten Legacy-Modus zu `-3.bit_length()` (in Python: `-(3.bit_length())` $\to$ ungültig), während der moderne Modus korrekt `(-3).bit_length()` lieferte. Durch den Umstieg auf `emit-operand` für Objekt (`dot`) und Sequenz (`aref`) in *beiden* Modi ist jetzt auch das Orakel konsistent.

### C. Sauberer Umgang mit Altlasten, Dateien und Prozessen
- **`cl-ppcre` komplett eliminiert**: Sowohl `comments` in `py.lisp` als auch `normalize-string` in `transpiler-tests.lisp` laufen jetzt mit reinem Common Lisp – eine Abhängigkeit weniger.
- **`write-source` & `write-notebook` gehärtet**:
  - `*file-hashes*` nutzt endlich `:test 'equal` mit dem Dateipfad als Schlüssel, prüft `(probe-file fn)` (gelöschte Dateien werden neu erstellt) und speichert den Zustand erst *nach* erfolgreichem Schreiben.
  - `write-notebook` legt keine feste `cl-py-generator-cell.py` mehr in `/tmp` ab, sondern formatiert jede Python-Zelle sauber über `uiop:with-temporary-file` im RAM/Temp-Scope.
- **`pipe.lisp` entkoppelt**: Keine undokumentierten `sb-impl::`-Interna mehr, sondern offizielles `sb-ext:process-status` und `sb-ext:process-pty` samt `nil`-EOF-Marker.
- **Dokumentation stimmt wieder**: `SUPPORTED_FORMS.md` hat kein peinliches `|"F.TXT"|` mehr, und `SKILL.md` erklärt die neue Readtable-Architektur verständlich.

---

## 2. Checkliste aller 12 Review-Punkte

| # | Befund aus dem Review | Status | Wie gelöst? |
|---|---|---|---|
| 1 | Globale `*readtable*`-Mutation | ✅ Gelöst | Entfernt; `py-symbol-name` + `named-readtables` lokal für Tests |
| 2 | Fehlende `cl-ppcre`-Abhängigkeit | ✅ Gelöst | `comment-newlines` & `normalize-string` ohne Regex |
| 3 | `paren*`-Fallback gibt `:left` aus | ✅ Gelöst | Vorab-Validierung; Legacy-Zweig nutzt nur `(third code)` |
| 4 | Defekter `:str`-Parameter in `emit-py` | ✅ Gelöst | Entfernt (kein Aufrufer nutzte ihn); `numberp` gibt Strings zurück |
| 5 | `\"f.txt\"` als Symbol in Tests | ✅ Gelöst | Auf `(open (string "f.txt"))` umgestellt + Doku regeneriert |
| 6 | Lücken in `parse-defun` | ✅ Gelöst | Rückgabetyp über `emit`, `&key` default `None`, leerer Body $\to$ `pass`, `&optional`/`&rest` implementiert, `captures` entfernt |
| 7 | `return` / `return_` & `nil` | ✅ Gelöst | 0 Args $\to$ `"return"`, 1 Arg (`nil` $\to$ `None`), >1 Args $\to$ Fehler |
| 8 | `write-source` & `*file-hashes*` | ✅ Gelöst | `equal`-Tabelle, `probe-file`-Check, Cache-Update nach Write |
| 9 | `write-notebook` Temp-Datei-Chaos | ✅ Gelöst | Zellweises In-Memory-Emit + `uiop:with-temporary-file` |
| 10 | `sb-impl::`-Interna in `pipe.lisp` | ✅ Gelöst | Umgestellt auf `sb-ext:process-status` / `sb-ext:process-pty` |
| 11 | Fehlende Operator-Arität (`:min-args`) | ✅ Gelöst | `:min-args 1` für n-äre/`-`, `:min-args 2` für Vergleiche/Shifts/`@` |
| 12 | Exports, Ruff-Warnungen, Fuzzer | ✅ Gelöst | `:not` exportiert, `:skip-ruff t` für Fragmente, Fuzzer erweitert |

---

## 3. Was bei meinem Gegen-Check noch aufgefallen ist

Auch wenn die 20 Tests in `issue-tests.lisp` und die bestehenden Suiten grün sind, zeigt ein genauer Blick in den geänderten Code noch **vier kleine Restpunkte**, von denen die ersten beiden wirklich behoben werden sollten:

### Fund 1: Halbherzige Paket-Unabhängigkeit in `write-notebook` (`py.lisp`)
In `notebook-cell-source` hat Muse Spark sehr mitgedacht und einen tollen Kommentar hinterlassen:
```lisp
;; The cell type is compared by name (upcased), so calls from any package
;; and any readtable case match: `python' works from CL-USER as well.
(case (intern (string-upcase (symbol-name name)) :cl-py-generator)
  (markdown ...)
  (python ...))
```
Schaut man aber 35 Zeilen tiefer in `write-notebook`, steht dort noch die alte Abfrage:
```lisp
(if (eq 'markdown (first e))
    (list (cons "cell_type" "markdown") ...)
    (list (cons "cell_type" "code") ...))
```
**Was passiert hier?** Ruft jemand `write-notebook` aus `CL-USER` mit `'((markdown "# Titel"))` auf, erkennt `notebook-cell-source` die Zelle zwar korrekt als Markdown – aber `write-notebook` vergleicht `eq 'cl-py-generator::markdown` mit `cl-user::markdown`, scheitert und verpackt den Markdown-Text als Python-Codezelle (`"cell_type": "code"`).  
**Fix (1 Zeile):** `(string-equal "markdown" (symbol-name (first e)))` statt `(eq 'markdown (first e))`.

---

### Fund 2: Kleiner Widerspruch zwischen Code und Docstring bei `py-symbol-name` (`py.lisp`)
Im Docstring von `py-symbol-name` steht ausdrücklich:
> *„Non-symbols are printed as-is.“*

Die Implementierung macht jedoch etwas anderes:
```lisp
(let ((name (if (symbolp sym)
                (symbol-name sym)
                (format nil "~a" sym))))
  (if (find-if #'lower-case-p name)
      name
      (string-downcase name)))
```
Wenn jemand bewusst einen rohen String ohne Kleinbuchstaben übergibt – etwa `(class "API" () ...)` oder `(import ("HTTP"))` –, wandelt `py-symbol-name` auch diesen String in `"api"` bzw. `"http"` um, obwohl es kein Symbol war.  
**Fix (2 Zeilen):**
```lisp
(if (symbolp sym)
    (let ((name (symbol-name sym)))
      (if (find-if #'lower-case-p name)
          name
          (string-downcase name)))
    (format nil "~a" sym))
```

---

### Fund 3: Ein Python-Stolperstein beim neuen `dot`-Fuzzer (`paren-tests.lisp`)
Im erweiterten `*random-forms*` steht jetzt:
```lisp
(lambda (sub) `(dot ,(funcall sub) (bit_length)))
```
Ist das Zufallsblatt eine **positive** Ganzzahl (z. B. `3`), emittieren *beide* Modi `3.bit_length()`. In Python ist das allerdings ein `SyntaxError`, weil der Parser `3.` als Gleitkommazahl `3.0` liest (korrekt wäre `(3).bit_length()`). Im Differentialtest fällt das nicht auf, weil beide Modi denselben `SyntaxError` werfen und Python ihn als identische `"EXCEPTION SyntaxError: ..."`-Strings vergleicht.  
Das ist kein Fehler von Muse Spark – die bestehende `emit-operand`-Regel klammerte nur negative Zahlen. Falls man `dot` auf positiven Integer-Literalen sauber unterstützen möchte, bräuchte `emit-operand` für `parent-op = 'dot` und `(integerp arg)` ebenfalls Klammern.

---

### Fund 4: Zwei Kleinigkeiten am Rande
1. **`lambda` vs. `parse-defun`**: Während `parse-defun` jetzt `&optional` und `&rest` sauber verarbeitet, ignoriert der `lambda`-Zweig in `emit-py` diese beiden weiterhin stillschweigend (war nicht direkt Teil der Aufgabe, fällt nur im Vergleich auf).
2. **`SKILL.md`-Hinweis**: Im Skill-Dokument steht unter *Language Architecture*, gemischte Namen würden nur erhalten bleiben, wenn die Datei `:cl-py-generator` via `named-readtables` aktiviert. Ein kurzer Zusatz, dass alternativ auch Standard-CL-Escaping wie `|Exception|` oder `|MyClass|` unter `:upcase` funktioniert, hilft Nutzern ohne `named-readtables`.

---

## 4. Gesamturteil & Empfehlung

Das Ergebnis auf `feature/hydrogen-review` ist **sehr überzeugend**:
- Alle 12 Review-Befunde wurden sauber adressiert und durch `issue-tests.lisp` abgesichert.
- Das `walkthrough.md` ist hervorragend geschrieben, nachvollziehbar und verschweigt keine Stolpersteine (wie die `E` $\to$ `e`-Umstellung in `def-type-annotations` oder den Orakel-Fix).
- Die beiden echten Restpunkte (**Fund 1** `eq 'markdown` in `write-notebook` und **Fund 2** Nicht-Symbole in `py-symbol-name`) sind in zwei Minuten behoben und sollten idealerweise noch mit zwei kurzen Tests in `issue-tests.lisp` abgesichert werden. Danach steht einem Merge nach `master` nichts mehr im Weg.


Hier sind die beiden neuen Tests im Format von `issue-tests.lisp` (direkt mit `define-issue-test` und `assert-true`, damit sie sich nahtlos in die bestehende Suite einfügen).

Beide schlagen auf dem aktuellen Stand von `py.lisp` fehl und werden grün, sobald Fund 1 und Fund 2 behoben sind.

```lisp
;;; ------------------------------------------------------------------
;;; Nachtrag aus dem Abnahme-Check (Fund 1 & Fund 2)
;;; ------------------------------------------------------------------

(define-issue-test "write-notebook-markdown-any-package"
  "write-notebook muss markdown-Zellen auch aus fremden Paketen (z. B. CL-USER) als cell_type \"markdown\" schreiben."
  (let* ((dir (uiop:ensure-directory-pathname
               (merge-pathnames "cl-py-gen-issue-md/" (uiop:temporary-directory))))
         (nb (merge-pathnames "md_pkg.ipynb" dir))
         (md-sym (intern "MARKDOWN" :cl-user)))
    (ensure-directories-exist dir)
    (unwind-protect
         (progn
           (write-notebook :nb-file (namestring nb)
                           :nb-code (list (list md-sym "# Titel")))
           (let ((json (uiop:read-file-string nb)))
             (assert-true (and (search "\"markdown\"" json)
                               (not (search "\"code\"" json)))
                          "Markdown-Zelle aus CL-USER wurde nicht als cell_type \"markdown\" geschrieben:~%~a"
                          json)))
      (uiop:delete-directory-tree dir :validate t :if-does-not-exist :ignore))))

(define-issue-test "py-symbol-name-preserves-non-symbols"
  "py-symbol-name darf Nicht-Symbole (wie rohe Strings in Grossbuchstaben) laut Docstring nicht kleinschreiben."
  (let ((direct (cl-py-generator::py-symbol-name "API"))
        (emitted (emit-py :clear-env t :code '(class "API" () pass))))
    (assert-true (string= "API" direct)
                 "py-symbol-name auf String \"API\" erwartet \"API\", erhalten: ~s"
                 direct)
    (assert-true (search "class API:" emitted)
                 "Klassenname aus rohem String \"API\" wurde kleingeschrieben: ~s"
                 emitted)))
```

### Kurzer Gegencheck zum Verhalten
- **`write-notebook-markdown-any-package`**:
  - *Aktuell*: `notebook-cell-source` akzeptiert `CL-USER::MARKDOWN`, aber `write-notebook` prüft `(eq 'markdown (first e))` und schreibt `"cell_type": "code"` ins JSON $\to$ **FAIL**. *(Der String-Check auf `"\"markdown\""` / `"\"code\""` funktioniert sowohl mit `jq` als auch ohne `jq`, weil die Notebook-Metadaten keines der beiden Wörter enthalten.)*
  - *Nach Fix* (`(string-equal "markdown" (symbol-name (first e)))` in `write-notebook`): schreibt `"cell_type": "markdown"` $\to$ **PASS**.
- **`py-symbol-name-preserves-non-symbols`**:
  - *Aktuell*: `(py-symbol-name "API")` liefert `"api"` und `(class "API" () pass)` emittiert `"class api:..."` $\to$ **FAIL**.
  - *Nach Fix* (Downcasing nur im `(symbolp sym)`-Zweig von `py-symbol-name`): liefert `"API"` bzw. `"class API:..."` $\to$ **PASS**.
