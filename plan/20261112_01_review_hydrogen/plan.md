# Plan: Hydrogen-Review umsetzen (Branch `feature/hydrogen-review`)

Stand: 2026-11-12. Ausgangspunkt: `master` (6284e0f). Ziel: alle Befunde aus
[`review.md`](plan/20261112_01_review_hydrogen/review.md) mit der
[`issue-tests.md`](plan/20261112_01_review_hydrogen/issue-tests.md)-Suite belegen und beheben.

## 1. Kontext und Architektur (für einen unabhängigen Agenten)

Der Transpiler `cl-py-generator` übersetzt S-Expressions nach Python. Zentrale
Dateien (alle im Repo-Root):

- [`py.lisp`](py.lisp) — der gesamte Emitter (`emit-py`, Operator-Tabellen
  `*infix-operators*`/`*precedence*`, `parse-defun`, `write-source`,
  `write-notebook`, `notebook-cell-source`). Fast alle Fixes landen hier.
- [`package.lisp`](package.lisp) — Paketdefinition, `:export`-Liste.
- [`pipe.lisp`](pipe.lisp) — SBCL-REPL-Anbindung (`start-python`, `run`).
- [`cl-py-generator.asd`](cl-py-generator.asd) — Systemdefinition (`:depends-on`).
- [`transpiler-tests.lisp`](transpiler-tests.lisp) — `*test-cases*` (`:name`,
  `:description`, `:lisp`, `:python`, `:tags`) plus Runner
  (`run-transpiler-tests`, `run-ruff-format`, `normalize-string`) und
  Doku-Generator (`generate-documentation` → [`SUPPORTED_FORMS.md`](SUPPORTED_FORMS.md)).
- [`paren-tests.lisp`](paren-tests.lisp) — Differentialtests: voll geklammerter
  Modus (`:omit-redundant-parentheses nil`, Orakel) gegen gekürzten Modus,
  beide Varianten werden von `python3` ausgewertet.
- [`issue-tests.lisp`](issue-tests.lisp) — NEU: 20 Regressionstests, einer pro
  Review-Befund (Format: `define-issue-test`, Runner `run-issue-tests`).
- [`SUPPORTED_FORMS.md`](SUPPORTED_FORMS.md) — generiert, nie von Hand editieren
  (vgl. [`AGENTS.md`](AGENTS.md)); wird per `./generate-docs.sh` neu erzeugt.

Wichtigste Erkenntnis aus der Analyse: Die globale
`(setf (readtable-case *readtable*) :invert)`-Zeile in `py.lisp` wird ersatzlos
gestrichen. Damit werden alle Dateien mit der Standard-`:upcase`-Readtable
gelesen (Symbole intern groß). Der Emitter bekommt dafür die neue Hilfsfunktion
`py-symbol-name`: Symbolnamen ohne Kleinbuchstaben werden per `string-downcase`
nach Python übertragen (`FOO` → `foo`, `:A` → `a`), gemischte Namen
(`|MyVar|`, `|Optional|`) bleiben erhalten. Das erhält kleingeschriebenes
Python, ohne fremde Sessions zu „verschmutzen“. Die Alternative
(`named-readtables`) würde alle `example/*/gen*.lisp` anfassen und ist daher
verworfen.

## 2. Lösungsentscheidungen pro Befund (in Review-Reihenfolge)

1. **Readtable-Mutation** — globale `setf`-Zeile löschen; `py-symbol-name`
   einführen und an allen Stellen nutzen, die Symbole als Python-Bezeichner
   drucken (Symbol-Zweig von `emit-py`, Funktions-/Klassen-/`def`-Namen,
   Keyword-Properties in Calls, `symbol`-Form, `import`-Namen).
2. **`cl-ppcre`-Abhängigkeit** — in `py.lisp` (`comments`: Zeilenumbruch
   ersetzen) und `transpiler-tests.lisp` (`normalize-string`: `\r\n?` → `\n`)
   durch Bordmittel ersetzen; kein neuer `:depends-on`-Eintrag nötig.
3. **`paren*`-Fallback** — Validierung (Argzahl, Position) vor die
   Modus-Verzweigung ziehen; Legacy-Zweig emittiert nur `(third code)`.
4. **`:str`-Parameter** — entfernen (kein Aufrufer in `example/` nutzt ihn);
   `numberp`-Zweig nutzt wieder `(format nil …)`.
5. **`"f.txt"`-Symbole** — `with-basic`/`with-as-basic` auf
   `(open (string "f.txt"))` umstellen (wie `with-multiple`); Doku neu
   generieren.
6. **`parse-defun`** — Rückgabetyp durch `emit` schicken; `&key`-Init `nil` →
   `None`; leerer Body → `(do (pass))`; `&optional` → `name`/`name=default`,
   `&rest` → `*name`; ungenutzte `captures`-Sammlung aus `consume-declare`
   entfernen.
7. **`return`/`return_`** — `return` direkt: 0 Args → `"return"`, 1 Arg →
   `"return <emit>"` (dabei `nil` → `None`), >1 Args → `error`. `return_`
   bleibt als Wrapper für Bestandstests erhalten.
8. **`write-source`** — `*file-hashes*` auf `:test 'equal`, Key
   `(namestring fn)`, Value `code-str`; nur bei
   `(and (probe-file fn) (string= old code-str))` überspringen; Eintrag erst
   nach erfolgreichem Schreiben setzen.
9. **`write-notebook`** — `notebook-cell-source` emittiert pro Python-Zelle im
   Speicher und formatiert genau einmal pro Zelle über
   `uiop:with-temporary-file`; keine feste Temp-Datei, keine
   `*file-hashes*`-Seiteneffekte mehr.
10. **`pipe.lisp`** — `sb-ext:process-status`/`sb-ext:process-pty`,
    `read-line … nil nil`.
11. **Arität** — `:min-args 1` für `+ * and or logand & logxor ^ logior | -`,
    `:min-args 2` für `< > <= >= == != << >> @` (`/ // % ** not ~ in/is`
    sind bereits gesetzt). `(+ x)`/`(or 255)` bleiben legal (Makro-Freundlichkeit).
12. **Paket/Doku/Fuzzer** — `:not` in `package.lisp` exportieren; fragmentarische
    Testfälle (`space-basic`, `for-generator-basic`) mit `:skip-ruff t`
    markieren und im Runner/Doku-Generator Ruff dafür überspringen;
    `*random-forms*` in `paren-tests.lisp` um `in`/`not-in`, `/`, `aref`,
    `dot`, `lambda`-Calls und 2-arg-`?` erweitern.

## 3. Commit-Plan (Conventional Commits, ein Commit pro Schritt)

1. `test(hydrogen): add issue-tests.lisp regression suite (red baseline)`
2. `fix(hydrogen): remove global readtable mutation, downcase emitted names`
3. `fix(hydrogen): drop cl-ppcre usage from emitter and test helper`
4. `fix(hydrogen): paren* legacy fallback, return arity, defun hardening`
5. `fix(hydrogen): write-source cache, write-notebook temp files, pipe sb-ext`
6. `fix(hydrogen): operator min-args, not export, test-case hygiene`
7. `docs(hydrogen): regenerate SUPPORTED_FORMS.md`
8. `docs(hydrogen): add walkthrough`

Jeder Commit: betroffene Tests laufen lassen (`issue-tests`, danach
`run-tests.sh`, `run-paren-tests.sh`).
