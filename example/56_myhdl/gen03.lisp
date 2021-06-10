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


	    (def led (sys_clk sys_rst_n led counter)
	      (do0
	       (@always clk.posedge sys_rst_n.negedge)
	       (def logic ()
		 (if (== rst 0)
		     (setf counter.next 0)
		     (setf counter.next (+ counter 1)))))
	      (return logic))
	    (setf counter (Signal (modbv 0 :min 0 :max ,(expt 2 24))))

	    (def test_led ()
	      (do0 
	       ,@(loop for e in `(sys_clk sys_rst_n)
		       collect
		       `(setf ,e (Signal (bool 0))))
	       ,@(loop for e in `(led)
		       collect
		       `(setf ,e (list (Signal (bool 0))
				       (Signal (bool 0))
				       (Signal (bool 0)))))
	       (setf inst (led sys_clk sys_rst_n led counter)))

	      (do0
	       (do0
		(@always (delay 10))
		(def clkgen ()
		  (setf sys_clk.next (not sys_clk))))

	       #+nil (do0
		(@always clk.negedge)
		(def stimulus ()
		  (setf d.next (randrange 2))))


	       (do0
		@instance
		(def rstgen ()
		  (yield (delay 5))
		  (setf sys_rst_n.next 0)
		  (while True
			 (do0 
			  (yield (delay (randrange 500 1000)))
			  (setf sys_rst_n.next 1))
			 (do0 
			  (yield (delay (randrange 80 140)))
			  (setf sys_rst_n.next 0)))))
	       (return (ntuple inst
			       clkgen
			       ;stimulus
			       rstgen))))

	    (def simulate (timesteps)
	      (setf tb (traceSignals test_led)
		    sim (Simulation tb))
	      (sim.run timesteps))
	    (simulate 2000)


	    (def convert ()
	      (do0 
	       ,@(loop for e in `(sys_clk sys_rst_n)
		       collect
		       `(setf ,e (Signal (bool 0))))
	       ,@(loop for e in `(led)
		       collect
		       `(setf ,e (list (Signal (bool 0))
				       (Signal (bool 0))
				       (Signal (bool 0)))))
	       (toVerilog led sys_clk sys_rst_n led counter))
	      )
	    (convert)
	    )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    (with-open-file (s (format nil "~a/~a" *source* "test.cst")
		       :direction :output
		       :if-exists :supersede :if-does-not-exist :create)
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



