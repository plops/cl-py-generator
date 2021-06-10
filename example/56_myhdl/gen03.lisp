(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/56_myhdl")
  (defparameter *code-file* "run_03_led")
  (defparameter *source* (format nil "~a/source/03_tang_led/" *path*))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	   `(do0
	     
	    (imports-from (myhdl *)
			  (random randrange))
	    (setf
	     _code_git_version
	     (string ,(let ((str (with-output-to-string (s)
				   (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			(subseq str 0 (1- (length str)))))
	     _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/run_00_flop.py")
				      )

	     _code_generation_time
	     (string ,(multiple-value-bind
			    (second minute hour date month year day-of-week dst-p tz)
			  (get-decoded-time)
			(declare (ignorable dst-p))
			(format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				hour
				minute
				second
				(nth day-of-week *day-names*)
				year
				month
				date
				(- tz))
			)))


	    (def led (sys_clk sys_rst_n led)
	      (do0
	       @always_comb ;; whenever on of the inputs changes
	       
	       (def logic ()
		 (if (== g 1)
		     (setf q.next d))))
	      (return logic))

	    (def test_latch ()
	      ,@(loop for e in `(q d g)
		      collect
		      `(setf ,e (Signal (bool 0))))
	      (setf inst (latch q d g))
	      ,@(loop for (q e f) in `((d dgen 7)
				       (g ggen 41))
		      collect
		      `(do0
		   (@always (delay ,f))
		   (def ,e ()
		     (setf (dot ,q next) (randrange 2)))))
	      (return (ntuple inst dgen ggen)))

	    (def simulate (timesteps)
	      (setf tb (traceSignals test_latch)
		    sim (Simulation tb))
	      (sim.run timesteps))
	    (simulate 2000)


	    (def convert ()
	      ,@(loop for e in `(q d g)
		      collect
		      `(setf ,e (Signal (bool 0))))
	      (toVerilog latch q d g))
	    (convert)
	    )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    (with-open-file (s (format nil "~a/~a" *source* "test.cst")
		       :direction :output)
      (flet ((p (str)
	       (format s "~a;~%" str)))
	(loop for (e f) in `((sys_clk 35)
			   (sys_rst_n 15)
			   ("led[0]" 16)
			   ("led[1]" 17)
			   ("led[2]" 18)
			   )
	      do
	      (p (format nil "IO_LOC \"~a\" ~a" e f)))))))



