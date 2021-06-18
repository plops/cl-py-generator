(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/56_myhdl")
  (defparameter *code-file* "run_06_lcpu")
  (defparameter *source* (format nil "~a/source/06_lcpu/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (let* ((code
	   `(do0
	     (imports-from (myhdl *)
			   
			)
	     (setf
	      _code_git_version
	      (string ,(let ((str (with-output-to-string (s)
				    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"))
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

	     (comments "https://hackaday.io/project/25686-lcpu-one-page-48-lines-myhdl-hardware-cpu")
	
	     (do0
	      @block
	      (def lcpu (iClk iRst oEXTMEM_addr oEXTMEM_data iEXTMEM_data o_EXTMEM_wr
			      PROGRAMM_MEM_CONTENT &key (CPU_BITWIDTH 8))
		(setf (tuple alu_iCODE
			     alu_iA
			     alu_iB
			     alu_RESULT)
		      (list (for-generator (i (range 4))
					   (aref (Signal (intbv 0))
						 (slice CPU_BITWIDTH "")))))
		(setf icode (aref (Signal (intbv 0)) (slice 8 ""))
		      reg_dst (list (for-generator (i (range 11))
						   (aref (Signal (intbv 0))
							 (slice CPU_BITWIDTH ""))))
		      src_data (list (for-generator (i (range 8))
						   (aref (Signal (intbv 0))
							 (slice CPU_BITWIDTH "")))))
		(setf alu_inst
		      (Alu.Alu alu_iCODE
			       alu_iA
			       alu_iB
			       alu_iCARRY
			       alu_RESULT
			       alu_CARRY
			       alu_ZERO))
		  
		  (do0
		   (@always_comb)
		   (def comb ()))
		(return comb)))

	     (do0
	      (def convert_this (hdl)
		(do0 
		    
		 (do0
		  (do0 ,@(loop for e in `((clk) (rst 1) (we))
			       collect
			       (destructuring-bind (name &optional (default 0)) e
				 `(setf ,name (Signal (bool ,default)))))
		       
		       ,@(loop for (e f) in `((adr 8) (di 8) (do 8))
			       collect
			       `(setf ,e (Signal (aref (modbv 0) (slice ,f "")))))
		       )
		  (setf mi (mem clk adr we di do)
			cpu (processor clk rst di do adr we))
		  (mi.convert :hdl hdl)
		  (cpu.convert :hdl hdl)
					;(toVerilog rom)
		  )))
	      (convert_this :hdl (string "Verilog"))))))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    (with-open-file (s (format nil "~a/~a" *source* "cpu_project.gprj")
		       :direction :output
		       :if-exists :supersede)
       ;; write project file for gowin ide
      (format s 
	      "<?xml version=\"1\" encoding=\"UTF-8\"?>
<!DOCTYPE gowin-fpga-project>
<Project>
    <Template>FPGA</Template>
    <Version>5</Version>
    <Device name=\"GW1N-1\" pn=\"GW1N-LV1QN48C6/I5\">gw1n1-004</Device>
    <FileList>
~{        ~a~^~%~}
    </FileList>
</Project>
"
	      (loop for (e f) in `(;(top.v verilog)
				   (mem.v verilog)
				   (processor.v verilog)
				   ;(osc.v verilog)
				   ;(rpll.v verilog)
				   (test.cst cst))
		    collect
		    (format nil "<File path=\"~a\" type=\"file.~a\" enable=\"1\"/>" e f))))

  
    (with-open-file (s (format nil "~a/~a" *source* "test.cst")
		       :direction :output
		       :if-exists :supersede :if-does-not-exist :create)
      (flet ((p (str)
	       (format s "~a;~%" str)))
	(loop for (e f) in `((clk 35)
			     (rst 14)
			     			     
			     )
	      do
		 (p (format nil "IO_LOC \"~a\" ~a" e f)))
	(loop for e in `((di (0 7) (27 28 29 30  31 32 33 34))
			 (do (0 7) (3 38 39 40  41 42 43 44))
			 (adr (0 7)(8 45 46 47  48 4 5 6)))
	      
	      do
		 (destructuring-bind (name (start end) vals) e
		   (loop for v in vals and i from start upto end
			 do
			    (p (format nil "IO_LOC \"~a[~a]\" ~a" name i v)))))
	
	(loop for e in `(rst clk)
	      do
		 (if (listp e)
		     (destructuring-bind (name i) e
		       (loop for count from 0 upto i do
			(p (format nil "IO_PORT \"~a[~a]\" IO_TYPE=LVCMOS33" name count))))
		     (p (format nil "IO_PORT \"~a\" IO_TYPE=LVCMOS33" e))))))))

