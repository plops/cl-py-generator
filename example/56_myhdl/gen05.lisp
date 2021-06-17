(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/56_myhdl")
  (defparameter *code-file* "run_05_8bit_cpu")
  (defparameter *source* (format nil "~a/source/05_8bit_cpu/" *path*))
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

	     (comments "https://nbviewer.jupyter.org/github/pcornier/1pCPU/blob/master/pCPU.ipynb")
	
	   
	     	(do0
		 @block
	      (def mem (clk adr we di do)
		(setf ram (list
			   (for-generator (i (range #x2000))
					  (Signal (aref (intbv 0) (slice 8 ""))))))
		(do0
		 (@always clk.posedge)
		 (def logic ()
		   (if we
		       (setf (dot (aref ram adr.val)
				  next)
			     di)
		       (if (< adr (len rom))
			   (setf do.next (aref rom adr.val))
			   (setf do.next (aref ram adr.val))))))
		(return logic)))
		 
		
	     (do0
	      (def convert_this (hdl)
		(do0 
		 
		 (do0
		  (do0 ,@(loop for e in `((clk) (rst 1) (we))
			       collect
			       (destructuring-bind (name &optional (default 0)) e
				`(setf ,name (Signal (bool ,default)))))
		       
		       ,@(loop for (e f) in `((adr 16) (di 8) (do 8))
			       collect
			       `(setf ,e (Signal (aref (modbv 0) (slice ,f "")))))
		       )
		  (setf mi (mem clk adr we di do))
		  (mi.convert :hdl hdl)
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
				   (cpu.v verilog)
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
	(loop for (e f) in `((xtal_in 35)
			     (n_rst 14)
			     ;(key 15)
			     ;(led_r 16) (led_g 17) (led_b 18)
			     )
	      do
		 (p (format nil "IO_LOC \"~a\" ~a" e f)))
	(loop for (e f) in `((clk 11)
			     (de 5)
			     (vsync 46)
			     (hsync 10)
			     )
	      do
		 (p (format nil "IO_LOC \"lcd_~a\" ~a" e f)))
	(loop for e in `((r (0 4) (27 28 29 30 31))
			 (g (0 5) (32 33 34 38 39 40))
			 (b (0 4) (41 42 43 44 45)))
	      
	      do
		 (destructuring-bind (name (start end) vals) e
		   (loop for v in vals and i from start upto end
			 do
			    (p (format nil "IO_LOC \"lcd_~a[~a]\" ~a" name i v)))))
	(loop for e in `(n_rst lcd_clk lcd_vsync lcd_de ;key
			       xtal_in lcd_hsync
			       (lcd_r 4) (lcd_g 5) (lcd_b 4))
	      do
		 (if (listp e)
		     (destructuring-bind (name i) e
		       (loop for count from 0 upto i do
			(p (format nil "IO_PORT \"~a[~a]\" IO_TYPE=LVCMOS33" name count))))
		     (p (format nil "IO_PORT \"~a\" IO_TYPE=LVCMOS33" e))))))))

