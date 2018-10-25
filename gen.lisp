(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))
;; https://www.kdnuggets.com/2018/08/introduction-t-sne-python.html
;; https://www.youtube.com/watch?v=RJVL80Gg3lA
;; sudo pacman -S autopep8

(in-package :cl-py-generator)

;; strace -f -s 256 -p `ps x|grep python3.5|head -n 1|cut -d " " -f 1`
(defparameter *python* 
  (sb-ext:run-program "python3" '()
		      :search t :wait nil
		      :pty t))

(defun run (code)
  (assert (eq :running (sb-impl::process-%status *python*)))
  (let ((s (sb-impl::process-pty *python*)))
    (write-sequence
    (cl-py-generator::emit-py  :clear-env t
			       :code code)
    s)
   (terpri s)
   (force-output s)))

(sb-thread:make-thread
 #'(lambda (standard-output)
     (let ((*standard-output* standard-output))
      (loop for line = (read-line (sb-impl::process-pty *python*) nil 'foo)
	 until (eq line 'foo)
	 do
	   (print line))))
 :name "python-reader"
 :arguments (list *standard-output*))



(let ((code
       `(do0
	 (imports (sys
		   (plt matplotlib.pyplot)
		   (np numpy)
		   (pd pandas)
		   pathlib))
	 (plt.ion)
	 (setf x (np.linspace 0 2.0 30)
	       y (np.sin x))
	 (plt.plot x y)
	 (plt.grid))))
  (run code)
  (write-source "/home/martin/stage/cl-py-generator/source/code" code)
  )
