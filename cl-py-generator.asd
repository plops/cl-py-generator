(asdf:defsystem cl-py-generator
    :version "0"
    :description "Emit Python code"
    :maintainer " <kielhorn.martin@gmail.com>"
    :author " <kielhorn.martin@gmail.com>"
    :licence "GPL"
    ;; external-program is not used by the library itself anymore (uiop is used
    ;; instead), but a few examples in example/ still call external-program:run.
    :depends-on ("alexandria" "jonathan" "uiop" "external-program")
    :serial t
    :components ((:file "package")
		 (:file "py")
		 ;(:file "transpiler-tests")
		 #+sbcl (:file "pipe")) )

