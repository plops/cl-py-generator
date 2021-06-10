(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/56_myhdl")
  (defparameter *code-file* "run_00_flop")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

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

	    (def dff (q d clk)
	      (do0
	       (@always clk.posedge)
	       (def logic ()
		 (setf q.next d)))
	      (return logic))

	    (def test_dff ()
	      (setf (ntuple q d clk)
		    (list
		     (for-generator (i (range 3))
				    (Signal (bool 0)))))
	      (setf dff_inst (dff q d clk))
	      (do0
	       (@always (delay 10))
	       (def clkgen ()
		 (setf clk.next (not clk))))

	      (do0
	       (@always clk.negedge)
	       (def stimulus ()
		 (setf d.next (randrange 2))))
	      (return (ntuple dff_inst
				 clkgen
				 stimulus)))

	    (def simulate (timesteps)
	      (setf tb (traceSignals test_dff)
		    sim (Simulation tb))
	      (sim.run timesteps))
	    (simulate 2000)


	    (def convert ()
	      ,@(loop for e in `(q d clk)
		      collect
		      `(setf ,e (Signal (bool 0))))
	      (toVerilog dff q d clk))
	    (convert)
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



