(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)

;(push :allop *features*)

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/59_nmigen")
  (defparameter *code-file* "run_01_mcpu")
  (defparameter *source* (format nil "~a/source/01_mcpu" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *py-modules*
    `(nmigen

      (pd pandas)
      (np numpy)
      ))
  (defparameter *cpu-ports* `((data 8)
			      (adr 6)
			      (oe)
			      (we)
			      (rst)
			      (clk)))
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

	      (def LCat (*args)
		(return (Cat (aref *args (slice "" "" -1)))))
	      
	      ,(let ((word-size 8)
		     (address-size 6))
		 `(class RamChip (Elaboratable)
		       (def __init__ (self)
			 
			 ,@(loop for e in `((data ,word-size)
					    (adr ,address-size)
					    (oe)
					    (we)
					    (cs))
				 collect
				 (destructuring-bind (name &optional (size 1)) e
				   `(setf (dot self ,name) (Signal ,size))))
			 (setf mem (Signal ,word-size))
			 )))
	      (class MCPU (Elaboratable)
		     (def __init__ (self )
		       ,@(loop for e in `(;; ports
					  ,@*cpu-ports* 
					  ;; registers
					  (accumulator 9)
					  (adreg 6)
					  (pc 6)
					  (states 3)
					  )
			       collect
			       (destructuring-bind (name &optional (size 1)) e
				`(setf (dot self ,name) (Signal ,size))))
		       
		       )
		     (def ports (self)
		       (return (list self.data
				     self.adr
				     self.oe
				     self.we
				     self.rst
				     self.clk))
		       )
		     (def elaborate (self platform)
		       (setf m (Module))
		       ,@(loop for code in `((self.adr.eq self.adreg)
					     (self.data.eq  	         (Mux (== self.states (C #b001 3)
								      )
								  (aref self.accumulator (slice 0 8))
								  0 ;; fixme: tristate
								  )
									 )
					     (self.oe.eq (logior self.clk
								 (~ self.rst)
								 (== self.states (C #b001 3))
								 (== self.states (C #b101 3))))
					     (self.we.eq (logior self.clk
								 (~ self.rst)
								 (!= self.states (C #b001 3))))
					     )
			       collect
			       `(incf m.d.comb
				      ,code))
		       (do0
			;; pc / address path
			(with (m.If (self.states.any))
				  (incf m.d.sync (self.adreg.eq self.pc))
				  
				  )
			    (with (m.Else )
				  (incf m.d.sync
					(self.pc.eq (+ self.adreg 1)))
				  (incf m.d.sync
					(self.adreg.eq (aref self.data (slice 0 6))))))
		       
		       (do0
			;; alu / data path
			(with (m.Switch self.states)
			      ,@(loop for (e f) in `((#b010 (self.accumulator.eq
							     ;; add
							     (+ (Cat 0
								     (aref self.accumulator
									   (slice 0 8)))
								(Cat 0 self.data))))
						     (#b011 (dot self (aref accumulator (slice 0 8))
								 (eq ;; nor
								  (~
								   (logior (aref self.accumulator (slice 0 7))
									   self.data)))))
						     (#b101 (dot self (aref accumulator 8)
								 ;; branch not taken, clear carry
								 (eq 0))))
				      collect
				      `(with (m.Case ,e)
					     ,f))))

		       (do0
			;; state machine
			(with (m.If (self.states.any))
			      (incf m.d.sync (self.states.eq 0)))
			(with (m.Else)
			      (with (m.If (logand (aref self.accumulator 8)
					       (dot  (aref self.data (slice 6 8))
						     (all))))
				    (incf m.d.sync (self.states.eq #b101)))
			      (with (m.Else)
				    (incf m.d.sync (self.states.eq (LCat 0 (~ (aref self.data (slice 6 8))))))))
			)
		       
		       
		 
		       (return m)))
	      )

	     ;; look at code without comments: cat mcpu.v |sed 's/\(*.*\)//g'|sed 's+/*.*/++g'|grep -v "^$"
	     #+nil (do0
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

	     
	     #+nil (do0
	      
	      (setf top (MCPU))
	      (with (as (open (string "mcpu.v")
			      (string "w"))
			f)
		    (f.write (verilog.convert top
					      :ports (list 
						      ,@(loop for e in `(data adr oe we rst clk)
							      collect
							      `(dot top ,e))
						      )))))

	     (do0
	      (setf m (Module))
	      (setf cpu (MCPU))
	      (setf m.submodules.cpu cpu)


	      (do0
	       ,@(loop for e in *cpu-ports*
			       collect
			       (destructuring-bind (name &optional (size 1)) e
				 `(do0 (setf ,name (Signal ,size))
				       (incf m.d.comb (dot cpu ,name (eq ,name)))))))
	      
	      (setf sim (Simulator m))
	      (def process ()
		(yield (dot oe (eq 0))))
	      (sim.add_process process)
	      (sim.add_clock 1e-6)
	      (with (sim.write_vcd (string "test.vcd")
				   (string "test.gtkw")
				   :traces (cpu.ports))
		    (sim.run)))

	     
	     )))
    (write-source (format nil "~a/~a" *source* *code-file*) code)))

