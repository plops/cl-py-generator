# Tasks: Hydrogen-Review (abarbeitbar für eine schwächere KI)

Jede Aufgabe: Datei öffnen, exakte Stelle suchen, Änderung einbauen, danach den
angegebenen Test laufen lassen. SBCL immer mit `--disable-debugger` starten
(vgl. lisp-dev-Skill). Vor jedem SBCL-Aufruf `(ql:register-local-projects)`
ausführen, damit `:cl-py-generator` gefunden wird.

## Block A — Regressionstests (rot)

- [ ] A1. Lege [`issue-tests.lisp`](../../issue-tests.lisp) im Repo-Root an:
      Inhalt exakt aus `issue-tests.md` (Abschnitt „`issue-tests.lisp`“),
      plus `(ql:register-local-projects)` vor dem ersten `ql:quickload`.
- [ ] A2. Starte
      `sbcl --noinform --disable-debugger --load issue-tests.lisp --eval '(cl-py-generator/issue-tests::run-issue-tests)' --quit`
      und halte fest: fast alle 20 Tests müssen FEHLSCHLAGEN (erwartet PASS nur
      bei `comments-multiline-string`). Abbruch bei Erfolg = Fehler im Setup.

## Block B — Emitter-Kern (`py.lisp`)

- [ ] B1. Lösche die Zeile `(setf (readtable-case *readtable*) :invert)`.
- [ ] B2. Neue Hilfsfunktion (nahe `join-strings`):
      `(defun py-symbol-name (sym) …)` — `symbol-name`, und wenn kein
      Kleinbuchstabe enthalten ist, `string-downcase`. Für Nicht-Symbole den
      Wert per `format nil "~a"` zurückgeben.
- [ ] B3. Ersetze jede Stelle, die ein Symbol als Python-Bezeichner druckt,
      durch `py-symbol-name`: Symbol-Zweig von `emit-py`
      (`(format nil "~a" code)`), Funktionsname im Call-Zweig, `def`-Name und
      Parameter in `parse-defun`, Klassenname, Keyword-Properties
      (`` `(= ,(format nil "~a" e) …)``), `symbol`-Form, `import`-Namen.
- [ ] B4. `comments`: ersetze `cl-ppcre:regex-replace-all` durch
      `with-output-to-string`-Schleife, die `#\Newline` durch Zeilenumbruch
      plus `"# "` ersetzt. Danach darf `py.lisp` den String `cl-ppcre:`
      nirgends mehr enthalten.
- [ ] B5. `paren*`: Argzahl-/Positionsprüfung vor die
      `omit-redundant-parentheses`-Verzweigung ziehen; Legacy-Zweig:
      `(format nil "(~a)" (emit (third code)))`.
- [ ] B6. `emit-py`: `&key`-Parameter `str` streichen; im `numberp`-Zweig
      `(format str …)` → `(format nil …)`.
- [ ] B7. `parse-defun`: Rückgabetyp `(car r)` → `(funcall emit (car r))`;
      Key-Init `nil` → `"= None"`; Body `(do ,@body)` → `(do ,@(or body '(pass)))`;
      `&optional`-Einträge `(name init _)` als `name` bzw. `name=init-oder-None`,
      `&rest`-Symbol als `*name` in die Parameterliste aufnehmen;
      `captures`-Block aus `consume-declare` (inkl. Docstring-Satz) entfernen.
- [ ] B8. `return`: 0 Args → `"return"`; 1 Arg → `"return <emit>"` wobei
      `nil` → `"None"`; mehr Args → `(error …)`. `return_` unverändert lassen.

## Block C — Dateien, Notebooks, Prozesse (`py.lisp`, `pipe.lisp`)

- [ ] C1. `*file-hashes*`: `(make-hash-table :test 'equal)`; in
      `write-source` Key `(namestring fn)`, Value `code-str`; Schreiben nur
      überspringen bei `(and (probe-file fn) (gethash …) (string= …))`;
      `setf gethash` erst NACH `with-open-file` + `format-python-file`.
- [ ] C2. `notebook-cell-source` (Python-Zweig): alle Formen der Zelle per
      `emit-py` in einen String sammeln, einmalig über
      `uiop:with-temporary-file` + `format-python-file` formatieren, danach in
      `"…\n"`-Zeilen aufspalten. Keine feste Datei
      `cl-py-generator-cell.py` mehr, kein `write-source`-Aufruf.
- [ ] C3. `pipe.lisp`: `sb-impl::process-%status` →
      `(sb-ext:process-status *python*)`, `sb-impl::process-pty` →
      `(sb-ext:process-pty *python*)`, `read-line … nil 'foo` → `nil nil`
      (Schleife `until (null line)`).

## Block D — Tabellen, Pakete, Testdaten

- [ ] D1. `*infix-operators*`: `:min-args 1` für `+ - * and or & logand ^ logxor
      | logior` ergänzen; `:min-args 2` für `< > <= >= == != << >> @`.
      Bestehende Einträge (`/ // % ** in not-in is is-not not ~`) anfassen nicht.
- [ ] D2. `package.lisp`: `:not` in `:export` ergänzen.
- [ ] D3. `transpiler-tests.lisp`: `(ql:quickload "cl-ppcre")` entfernen;
      `normalize-string` ohne `cl-ppcre` (Schleife: `#\Return` ggf. mit
      folgendem `#\Newline` zu `#\Newline`); `with-basic`/`with-as-basic` auf
      `(open (string "f.txt"))` umstellen; `:skip-ruff t` an `space-basic` und
      `for-generator-basic` (ggf. weitere, falls `run-ruff-format` warnt);
      `run-transpiler-tests` und `generate-documentation` überspringen Ruff
      für `:skip-ruff`-Fälle (Rohvergleich).
- [ ] D4. `paren-tests.lisp`: `*random-forms*` erweitern um `in`/`not-in` (mit
      kleiner fester Liste, z. B. `(list 0 1)`), `/`, `aref` auf einer
      Listen-Expression, `dot` (z. B. `bit_length`-analog: Ganzzahl-Methode),
      `lambda`-Aufruf und 2-arg-`?`. Nur ganzzahlige, totale Formen verwenden.

## Block E — Validierung und Abschluss

- [ ] E1. `issue-tests.lisp` erneut laufen lassen: 20/20 PASS.
- [ ] E2. `./run-tests.sh` (SBCL + `ruff format` + `python3` nötig): grün.
- [ ] E3. `./run-paren-tests.sh`: grün (ggf. `:count 2000` für den Abschlusslauf).
- [ ] E4. `./generate-docs.sh`: [`SUPPORTED_FORMS.md`](../../SUPPORTED_FORMS.md)
      neu erzeugen, Diff prüfen (u. a. kein `|"F.TXT"|` mehr).
- [ ] E5. `walkthrough.md` (deutsch, didaktisch, mit Glossar, Beispielen,
      Mermaid-Diagrammen, Kapiteln 1–4 laut `prompt.txt`) schreiben; neue
      Container-Pakete auflisten (`sbcl`, `ruff`, `python3`, `jq` — falls für
      Tests nachinstalliert, sonst Bestand bestätigen).
- [ ] E6. Commits nach Commit-Plan aus `plan.md` erstellen (Conventional
      Commits). Regel: Lässt sich ein Problem in zwei Versuchen nicht lösen →
      anhalten und berichten, kein dritter Ansatz.
