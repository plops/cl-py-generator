(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/51_django/")

  (write-source
   (format nil "~a/mysite/polls/views.py" *path*)
   `(do0
     (imports (django.http))
     (def index (request)
       (return (django.http.HttpResponse (string "hello world")))))))



