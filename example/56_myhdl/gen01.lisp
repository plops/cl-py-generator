(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/56_myhdl")
  (defparameter *code-file* "run_01_dffa")
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

	    (comments "d flip-flop with asynchronous reset")

	    (def dffa (q d clk rst)
	      (do0
	       (@always clk.posedge rst.negedge)
	       (def logic ()
		 (if (== rst 0)
		     (setf q.next 0)
		     (setf q.next d))))
	      (return logic))

	    (def test_dffa ()
	      ,@(loop for e in `(q d clk rst)
		      collect
		      `(setf ,e (Signal (bool 0))))
	      (setf dffa_inst (dffa q d clk rst))
	      (do0
	       (@always (delay 10))
	       (def clkgen ()
		 (setf clk.next (not clk))))

	      (do0
	       (@always clk.negedge)
	       (def stimulus ()
		 (setf d.next (randrange 2))))


	      (do0
	       @instance
	       (def rstgen ()
		 (yield (delay 5))
		 (setf rst.next 1)
		 (while True
			(do0 
			 (yield (delay (randrange 500 1000)))
			 (setf rst.next 0))
			(do0 
			 (yield (delay (randrange 80 140)))
			 (setf rst.next 1)))))
	      (return (ntuple dffa_inst
			      clkgen
			      stimulus
			      rstgen)))

	    (def simulate (timesteps)
	      (setf tb (traceSignals test_dffa)
		    sim (Simulation tb))
	      (sim.run timesteps))
	    (simulate 2000)


	    (def convert ()
	      ,@(loop for e in `(q d clk rst)
		      collect
		      `(setf ,e (Signal (bool 0))))
	      (toVerilog dffa q d clk rst))
	    (convert)
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



