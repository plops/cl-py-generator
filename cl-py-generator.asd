(asdf:defsystem cl-py-generator
    :version "0"
    :description "Emit Python code"
    :maintainer " <kielhorn.martin@gmail.com>"
    :author " <kielhorn.martin@gmail.com>"
    :licence "GPL"
    :depends-on ("alexandria" "jonathan" "external-program")
    :serial t
    :components ((:file "package")
		 (:file "py")
		 ;(:file "transpiler-tests")
		 #+sbcl (:file "pipe")) )
x
