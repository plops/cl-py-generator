(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)

;(push :allop *features*)

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/59_nmigen")
  (defparameter *code-file* "run_00_first")
  (defparameter *source* (format nil "~a/source/00_first" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *py-modules*
    `(nmigen
      (pd pandas)
      (np numpy)
      ))
  (let* ((code
	   `(do0
	     (imports (sys))
	     (imports ,*py-modules*)
	     (imports-from (nmigen  *)
			   (nmigen.back.pysim *)
			   (nmigen.back verilog)
			   (nmigen.utils *)
			   ;(nmigen.compat.sim Simulator)
			   )
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/58_migen/source/00_first/run_00.py"))
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
				 (- tz)))))
	     (do0
	      (setf res (list ))
	      (for ((ntuple idx s) (enumerate (dot sys.version (split (string "\\n")))))
		   (res.append (dictionary :key (dot (string "python_{}")
						     (format idx))
					   :value s)))
	      ,@(loop for e in *py-modules*
		      collect
		      (let ((short-name e)
			    (name e))
			(when (listp e)
			  (setf short-name (first e)
				name (second e)))
			`(res.append (dictionary :key (string ,name)
						 :value (dot ,short-name __version__)
						 ))))
	      (setf df_status (pd.DataFrame res))
	      (print df_status))

	     (do0

	      ;; https://nmigen.info/nmigen/latest/start.html
	      (class UpCounter (Elaboratable)
		     (def __init__ (self limit)
		       (setf self.limit limit
			     ;; ports
			     self.en (Signal)
			     self.ovf (Signal)
			     ;; state
			     self.count (Signal (range 0 (+ 1 limit))
						#+nil(int (np.ceil (/ (np.log limit)
								    (np.log 2)))))
			     )
		       )
		     (def elaborate (self platform)
		       (setf m (Module))
		       (incf m.d.comb
			     (self.ovf.eq (== self.count
					      self.limit)))
		       (with (m.If self.en)
			     (with (m.If self.ovf)
				   (incf m.d.sync
					 (self.count.eq 0)))
			     (with (m.Else)
				   (incf m.d.sync
					 (self.count.eq (+ self.count 1)))))
		       (return m)))
	      )

	     (do0
	      (setf dut (UpCounter 25))
	      (def bench ()
		(yield (dut.en.eq 0))
		(for (_ (range 30))
		     yield
		     (assert (not (yield dut.ovf))))
		(yield (dut.en.eq 1))
		(for (_ (range 25))
		     yield
		     (assert (not (yield dut.ovf))))
		yield
		(assert (yield dut.ovf))
		yield
		(assert (not (yield dut.ovf))))
	      (do0
	       (setf sim (Simulator dut)
		     )
	       (sim.add_clock 1e-6)
	       (sim.add_sync_process bench)
	       (with (sim.write_vcd (string "up_counter.vcd"))
		     (sim.run))))

	     (do0
	      
	      (setf top (UpCounter 25))
	      (with (as (open (string "up_counter.v")
			      (string "w"))
			f)
		    (f.write (verilog.convert top
					      :ports (list top.en
							   top.ovf)))))
	     )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)))

