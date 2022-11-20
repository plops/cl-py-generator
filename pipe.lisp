(in-package :cl-py-generator)

;; strace -f -s 256 -p `ps x|grep python3.5|head -n 1|cut -d " " -f 1`
(defparameter *python* nil)
(defparameter *python-reading-thread* nil)


(defun start-python ()
  (unless (and *python*
	       (eq :running (sb-impl::process-%status *python*)))
    (setf *python*
	  (sb-ext:run-program "python3" '()
			      :search t :wait nil
			      :pty t))
    (when *python-reading-thread*
      (sb-thread:join-thread *python-reading-thread*)
      (setf *python-reading-thread* nil))
    (setf *python-reading-thread*
	  (sb-thread:make-thread
	   #'(lambda (standard-output)
	       (let ((*standard-output* standard-output))
		 (loop for line = (read-line (sb-impl::process-pty *python*) nil 'foo)
		       until (eq line 'foo)
		       do
		       (print line))))
	   :name "python-reader"
	   :arguments (list *standard-output*)))))

(defun run (code)
  (assert (eq :running (sb-impl::process-%status *python*)))
  (let ((s (sb-impl::process-pty *python*)))
    (write-sequence
     (cl-py-generator::emit-py  :clear-env t
				:code code)
     s)
    (terpri s)
    (force-output s)))


