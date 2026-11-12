;;;; issue-tests.lisp --- Regressionstests für die im Code-Review gefundenen Issues
;;;;
;;;; Aufruf:
;;;;   sbcl --noinform --disable-debugger --load issue-tests.lisp \
;;;;        --eval '(cl-py-generator/issue-tests::run-issue-tests)' --quit

(ql:register-local-projects)
(ql:quickload :cl-py-generator)
(ql:quickload "uiop")
(ql:quickload "asdf")

;; transpiler-tests.lisp laden, damit *test-cases* inspiziert werden kann
(let ((tests-file (merge-pathnames "transpiler-tests.lisp"
                                   (asdf:system-source-directory :cl-py-generator))))
  (when (probe-file tests-file)
    (load tests-file)))

(defpackage :cl-py-generator/issue-tests
  (:use :cl :cl-py-generator))

(in-package :cl-py-generator/issue-tests)

(defparameter *issue-tests* nil)

(defmacro define-issue-test (name description &body body)
  `(push (list :name ,name
               :description ,description
               :fn (lambda () ,@body))
         *issue-tests*))

(defun assert-true (cond fmt &rest args)
  (unless cond
    (apply #'error fmt args)))

(defun system-file (relative-path)
  (merge-pathnames relative-path (asdf:system-source-directory :cl-py-generator)))

;;; ------------------------------------------------------------------
;;; 1. Globale Mutation von *readtable* & Standard-Upcase-Verträglichkeit
;;; ------------------------------------------------------------------

(define-issue-test "readtable-not-mutated"
  "Das Laden von :cl-py-generator darf das globale *readtable* nicht auf :invert umstellen."
  (let ((*readtable* (copy-readtable nil)))
    (asdf:load-system :cl-py-generator :force t)
    (assert-true (eq :upcase (readtable-case *readtable*))
                 "Erwartet :UPCASE nach Laden des Systems, erhalten: ~s"
                 (readtable-case *readtable*))))

(define-issue-test "readtable-upcase-keywords"
  "Formen, die mit Standard-:upcase-Readtable gelesen werden, muessen klein geschriebene Python-Keywords liefern."
  (let* ((*readtable* (copy-readtable nil))
         (form (read-from-string "(cl-py-generator:dictionary :a 1)")))
    (let ((got (emit-py :clear-env t :code form)))
      (assert-true (string= "dict(a=1)" got)
                   "Erwartet \"dict(a=1)\", erhalten: ~s" got))))

;;; ------------------------------------------------------------------
;;; 2. Fehlende cl-ppcre-Abhängigkeit in .asd
;;; ------------------------------------------------------------------

(define-issue-test "asd-cl-ppcre-dependency"
  "Wenn py.lisp cl-ppcre referenziert, muss cl-ppcre in :depends-on der .asd-Datei stehen."
  (let* ((py-src (uiop:read-file-string (system-file "py.lisp")))
         (uses-ppcre (search "cl-ppcre:" py-src :test #'char-equal))
         (deps (asdf:system-depends-on (asdf:find-system :cl-py-generator)))
         (declares-ppcre (member "cl-ppcre" deps :test #'string-equal)))
    (assert-true (or (not uses-ppcre) declares-ppcre)
                 "py.lisp verwendet cl-ppcre, aber .asd :depends-on ist ~s" deps)))

(define-issue-test "comments-multiline-string"
  "Ein einzelnes Argument mit Newline in (comments ...) muss jede Zeile kommentieren."
  (let ((got (emit-py :clear-env t :code `(comments ,(format nil "line1~%line2")))))
    (assert-true (string= (format nil "# line1~%# line2~%") got)
                 "Erwartet mehrzeiligen Kommentar, erhalten: ~s" got)))

;;; ------------------------------------------------------------------
;;; 3. Bug im Fallback-Zweig von paren*
;;; ------------------------------------------------------------------

(define-issue-test "paren-star-legacy-position"
  "paren* mit Position darf im Legacy-Modus (:omit-redundant-parentheses nil) die Position nicht emittieren."
  (let ((got (emit-py :clear-env t
                      :code '(paren* * (+ a b) :left)
                      :omit-redundant-parentheses nil)))
    (assert-true (string= "(((a)+(b)))" got)
                 "Erwartet \"(((a)+(b)))\", erhalten: ~s" got))
  (assert-true (handler-case
                   (progn
                     (emit-py :clear-env t
                              :code '(paren* * (+ a b) :bogus)
                              :omit-redundant-parentheses nil)
                     nil)
                 (error () t))
               "Ungueltige Position bei paren* muss auch im Legacy-Modus einen Fehler werfen"))

;;; ------------------------------------------------------------------
;;; 4. Defekter :str-Parameter in emit-py
;;; ------------------------------------------------------------------

(define-issue-test "emit-py-str-parameter"
  "emit-py muss entweder :str sauber fuer zusammengesetzte Formen unterstuetzen oder :str entfernt haben."
  (let ((ok (handler-case
                (let ((written (with-output-to-string (s)
                                 (emit-py :clear-env t :code '(+ 1 2) :str s))))
                  (string= "1+2" written))
              ;; Falls :str bewusst aus der Signatur entfernt wurde:
              (program-error () t))))
    (assert-true ok
                 "emit-py akzeptiert :str, hat aber bei (+ 1 2) nichts in den Stream geschrieben")))

;;; ------------------------------------------------------------------
;;; 5. Fehlerhafte Symbol-Syntax "f.txt" in transpiler-tests.lisp
;;; ------------------------------------------------------------------

(define-issue-test "test-cases-no-quoted-symbols"
  "*test-cases* darf keine Symbole mit eingebetteten Anfuehrungszeichen wie |\"F.TXT\"| enthalten."
  (labels ((find-bad-symbol (node)
             (cond
               ((symbolp node)
                (when (find #\" (symbol-name node))
                  node))
               ((consp node)
                (or (find-bad-symbol (car node))
                    (find-bad-symbol (cdr node)))))))
    (dolist (tc cl-py-generator/tests::*test-cases*)
      (let ((bad (find-bad-symbol (getf tc :lisp))))
        (assert-true (null bad)
                     "Test ~s enthaelt Symbol mit Quote im Namen: ~s"
                     (getf tc :name) bad)))))

;;; ------------------------------------------------------------------
;;; 6. Lücken und Bugs in parse-defun
;;; ------------------------------------------------------------------

(define-issue-test "parse-defun-return-type-emitted"
  "Zusammengesetzte Rueckgabetypen in (declare (values ...)) muessen durch emit laufen."
  (let ((got (emit-py :clear-env t
                      :code '(def foo (x)
                               (declare (type (aref list int) x)
                                        (values (aref list int)))
                               (return x)))))
    (assert-true (and (not (search "(aref" got :test #'char-equal))
                      (search "list[int]" got))
                 "Rueckgabetyp wurde nicht korrekt emittiert: ~s" got)))

(define-issue-test "parse-defun-key-default-nil"
  "Keyword-Parameter ohne expliziten Init-Wert (&key (y)) muessen gueltiges Python (= None) erzeugen."
  (let ((got (emit-py :clear-env t
                      :code '(def foo (x &key (y))
                               (return y)))))
    (assert-true (and (not (search "= )" got))
                      (search "None" got))
                 "Ungueltige Default-Syntax bei &key (y): ~s" got)))

(define-issue-test "parse-defun-empty-body"
  "Eine Funktionsdefinition ohne Body muss gueltiges Python (pass) erzeugen."
  (let ((got (emit-py :clear-env t :code '(def foo ()))))
    (assert-true (search "pass" got)
                 "Leerer Funktionsbody enthaelt kein pass: ~s" got)))

(define-issue-test "parse-defun-optional-and-rest"
  "&optional und &rest duerfen nicht stillschweigend verworfen werden."
  (let ((ok (handler-case
                (let ((got (emit-py :clear-env t
                                    :code '(def foo (a &optional b &rest c)
                                             (return a)))))
                  (and (search "b" got) (search "*c" got)))
              ;; Alternativ expliziter Fehler statt stillem Verwerfen:
              (error () t))))
    (assert-true ok
                 "&optional / &rest wurden stillschweigend aus der Parameterliste verworfen")))

;;; ------------------------------------------------------------------
;;; 7. return / return_ und nil
;;; ------------------------------------------------------------------

(define-issue-test "return-multiple-args-error"
  "(return x y) darf das zweite Argument nicht stillschweigend verwerfen."
  (assert-true (handler-case
                   (progn (emit-py :clear-env t :code '(return x y)) nil)
                 (error () t))
               "(return x y) hat keinen Fehler geworfen und y verschluckt"))

(define-issue-test "return-empty-and-nil"
  "(return) muss 'return' ohne Trailing Space und (return nil) 'return None' emittieren."
  (let ((empty-ret (emit-py :clear-env t :code '(return)))
        (nil-ret (emit-py :clear-env t :code '(return nil))))
    (assert-true (string= "return" empty-ret)
                 "(return) erwartet \"return\", erhalten: ~s" empty-ret)
    (assert-true (string= "return None" nil-ret)
                 "(return nil) erwartet \"return None\", erhalten: ~s" nil-ret)))

;;; ------------------------------------------------------------------
;;; 8. write-source und *file-hashes*
;;; ------------------------------------------------------------------

(define-issue-test "write-source-recreates-deleted-file"
  "write-source muss eine auf Platte geloeschte Zieldatei trotz Hash-Cache neu erzeugen."
  (let* ((dir (uiop:ensure-directory-pathname
               (merge-pathnames "cl-py-gen-issue8a/" (uiop:temporary-directory))))
         (name "recreate_test")
         (fn (merge-pathnames (format nil "~a.py" name) dir)))
    (ensure-directories-exist dir)
    (unwind-protect
         (progn
           (when (probe-file fn) (delete-file fn))
           (write-source name '(setf x 1) dir)
           (assert-true (probe-file fn) "Erstes write-source hat Datei nicht erstellt")
           (delete-file fn)
           (write-source name '(setf x 1) dir)
           (assert-true (probe-file fn)
                        "Zweites write-source nach Loeschen hat Datei wegen *file-hashes* uebersprungen"))
      (uiop:delete-directory-tree dir :validate t :if-does-not-exist :ignore))))

(define-issue-test "write-source-error-no-cache-poison"
  "Ein fehlgeschlagenes write-source darf *file-hashes* nicht vorab befuellen."
  (let* ((base (uiop:ensure-directory-pathname
                (merge-pathnames "cl-py-gen-issue8b/" (uiop:temporary-directory))))
         (sub (merge-pathnames "missing_dir/" base))
         (name "poison_test")
         (fn (merge-pathnames (format nil "~a.py" name) sub)))
    (uiop:delete-directory-tree base :validate t :if-does-not-exist :ignore)
    (unwind-protect
         (progn
           (ignore-errors (write-source name '(setf x 42) sub))
           (ensure-directories-exist sub)
           (write-source name '(setf x 42) sub)
           (assert-true (probe-file fn)
                        "Fehlgeschlagenes write-source hat *file-hashes* vergiftet"))
      (uiop:delete-directory-tree base :validate t :if-does-not-exist :ignore))))

;;; ------------------------------------------------------------------
;;; 9. write-notebook Temp-Dateien
;;; ------------------------------------------------------------------

(define-issue-test "write-notebook-no-stray-temp-file"
  "write-notebook darf keine feste Datei cl-py-generator-cell.py im Temp-Verzeichnis hinterlassen."
  (let* ((dir (uiop:ensure-directory-pathname
               (merge-pathnames "cl-py-gen-issue9/" (uiop:temporary-directory))))
         (nb1 (merge-pathnames "nb1.ipynb" dir))
         (nb2 (merge-pathnames "nb2.ipynb" dir))
         (stray (merge-pathnames "cl-py-generator-cell.py" (uiop:temporary-directory))))
    (ensure-directories-exist dir)
    (unwind-protect
         (progn
           (when (probe-file stray) (delete-file stray))
           (write-notebook :nb-file (namestring nb1)
                           :nb-code '((python (setf nb_test 1))))
           (assert-true (not (probe-file stray))
                        "write-notebook hat feste Temp-Datei ~a hinterlassen" stray)
           ;; Selbst wenn Temp-Dateien geloescht wurden, muss ein zweiter Aufruf klappen
           (write-notebook :nb-file (namestring nb2)
                           :nb-code '((python (setf nb_test 1))))
           (assert-true (probe-file nb2) "Zweites Notebook wurde nicht erzeugt"))
      (uiop:delete-directory-tree dir :validate t :if-does-not-exist :ignore)
      (when (probe-file stray) (delete-file stray)))))

;;; ------------------------------------------------------------------
;;; 10. SBCL-Interna in pipe.lisp
;;; ------------------------------------------------------------------

(define-issue-test "pipe-no-sb-impl-internals"
  "pipe.lisp darf keine unexportierten sb-impl::-Interna verwenden."
  (let ((src (uiop:read-file-string (system-file "pipe.lisp"))))
    (assert-true (not (search "sb-impl::" src :test #'char-equal))
                 "pipe.lisp enthaelt sb-impl::-Aufrufe (statt sb-ext:)")))

;;; ------------------------------------------------------------------
;;; 11. Operator-Arity-Prüfungen
;;; ------------------------------------------------------------------

(define-issue-test "operator-arity-checks"
  "Operatoren mit zu wenig Argumenten (0-stelliges -/+/* oder 1-stellige Vergleiche/Shifts) muessen Fehler werfen."
  (dolist (bad-form '((-) (+) (*) (== a) (< a) (<< a)))
    (assert-true (handler-case
                     (progn (emit-py :clear-env t :code bad-form) nil)
                   (error () t))
                 "Erwarteter Arity-Fehler fuer ~s" bad-form)))

;;; ------------------------------------------------------------------
;;; 12. Package-Exports & Ruff-Warnungen in transpiler-tests.lisp
;;; ------------------------------------------------------------------

(define-issue-test "package-exports-not"
  ":cl-py-generator muss 'not' genau wie 'and' und 'or' exportieren."
  (multiple-value-bind (sym status)
      (find-symbol (symbol-name 'cl:not) :cl-py-generator)
    (declare (ignore sym))
    (assert-true (eq :external status)
                 "Symbol-Status von NOT in :cl-py-generator ist ~s (erwartet :EXTERNAL)"
                 status)))

(define-issue-test "transpiler-tests-no-ruff-warnings"
  "Formatieren aller *test-cases* darf keine Ruff-Warnungen (z. B. durch nicht-eigenstaendige Fragmente) ausloesen."
  (let ((warnings nil))
    (handler-bind ((warning (lambda (w)
                              (push (princ-to-string w) warnings)
                              (muffle-warning w))))
      (dolist (tc cl-py-generator/tests::*test-cases*)
        (unless (getf tc :skip-ruff)
          (cl-py-generator/tests::run-ruff-format
           (emit-py :clear-env t :code (getf tc :lisp))))))
    (assert-true (null warnings)
                 "Ruff-Formatierung hat ~d Warnung(en) erzeugt, u. a.: ~a"
                 (length warnings) (first warnings))))

;;; ------------------------------------------------------------------
;;; Runner
;;; ------------------------------------------------------------------

(defun run-issue-tests ()
  "Fuehrt alle Issue-Regressionstests aus und beendet bei Fehlern mit Exit-Code 1."
  (let ((passed 0)
        (failed 0)
        (tests (reverse *issue-tests*)))
    (format t "~&Running ~d issue tests...~%" (length tests))
    (dolist (entry tests)
      (let ((name (getf entry :name))
            (fn (getf entry :fn)))
        (handler-case
            (progn
              (funcall fn)
              (incf passed)
              (format t "  [PASS] ~a~%" name))
          (error (e)
            (incf failed)
            (format t "  [FAIL] ~a: ~a~%" name e)))))
    (format t "~&--- Issue Test Summary ---~%")
    (format t "Passed: ~d, Failed: ~d~%" passed failed)
    (format t "--------------------------~%")
    (unless (zerop failed)
      (uiop:quit 1))
    failed))
