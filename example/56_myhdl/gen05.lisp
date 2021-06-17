(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)

(push :allop *features*)

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
			   (collections namedtuple)
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
	

	     ,(let ((adm-l `(imp imm abs rel idx))
		    (opc-l `(lda sta
					;
				 pha pla asl asr
				 txa tax inx
					;
				 dex add sub and or xor
				 cmp
					;
				 rts
				 jnz
					;
				 jz jsr jmp
				 )))
		`(do0
		  ,@(loop for (name l) in `((adm ,adm-l)
					    (opc ,opc-l))
			  collect
			  `(do0
			    (setf ,(string-upcase name)
				  (namedtuple (string ,name)
						    (list ,@(loop for e in l collect
								  `(string ,(string-upcase e))))
						    ))
			    (setf ,name (,(string-upcase name)
					 (*range ,(length l))))))))
	     
	     ,@(let ((ram-size #x100))
		 `( (setf rom (tuple ,@(loop for e in `(01 00 38 0c 00 01 40 30 79 10 8b f8 00)
					    collect
					    (format nil "0x~a" e))))
		   (do0
		    @block
		    (def mem (clk adr we di do)
		      (setf ram (list
				 (for-generator (i (range ,ram-size))
						(Signal (aref (intbv 0) (slice 8 ""))))))
		     
		      (do0
		       (@always clk.posedge)
		       (def logic ()
			 (if we
			     (setf (dot (aref ram adr.val)
					next)
				   di)
			     (if (< adr (len rom))
				 (setf do.next 0; (aref rom adr.val)
				       )
				 (setf do.next (aref ram adr.val))))))
		      (return logic)))

		    ,(let ((l `((ir instruction-register)
				(im immediate-value)
				(ra accumulator)
				(rx x-register)
				(rw w-register-for-status-flags)
				(sr status-register)
				(am addressing-mode)
				(sp stack-pointer))))
		      `(do0
			@block
			(def processor (clk rst di do adr we)

			  (setf (ntuple F1 F2 D E M1 M2) (range 0 6)
				pc (Signal (aref (modbv 0) "11:"))
				cyc (Signal (aref (modbv 0) "3:"))
				(ntuple ,@ (mapcar #'first l))
				(list (for-generator (i (range ,(length l)))
						      (Signal (aref (modbv 0) "8:"))))
				sp (Signal (aref (modbv #xff) "8:")))
			 (do0
		       (@always clk.posedge)
		       (def logic ()
			 (cond (rst
				 (setf pc.next 0
				       adr.next 0))
			       ((== cyc F1)
				(setf adr.next (+ pc 1)
				      pc.next (+ pc 1)
				      cyc.next F2))
			       ((== cyc F2)
				(setf adr.next (+ pc 1)
				      ir.next do
				      cyc.next D))
			       ((== cyc D)
				(setf im.next do
				      am.next (& ir 7)
				      ir.next (& (>> ir 3) #x1f))
				#+allop (when (== (>> ir 3) opc.RTS)
				  (setf adr.next (+ sp 1)
					sp.next (+ sp 1))
				  )
				(setf cyc.next E))
			       ((== cyc E)
				(cond ((== ir opc.LDA)
				       (cond ((== am adm.IMM)
					      (setf ra.next im ;; fixme: adr?
						    pc.next (+ pc 1)))
					     ((== am adm.ABS)
					      (setf adr.next (logior (<< do 8)
								     im)
						    pc.next (+ pc 2)))
					     ((== am adm.IDX)
					      (setf adr.next (logior (<< do 8)
								     (+ im rx))
						    pc.next (+ pc 2)))))
				       ((== ir opc.STA)
				       (cond 
					 ((== am adm.ABS)
					  (setf adr.next (logior (<< do 8)
								 im)
						we.next 1
						di.next ra
						pc.next (+ pc 2)))
					 ((== am adm.IDX)
					  (setf adr.next (logior (<< do 8)
								 (+ im rx))
						we.next 1
						di.next ra
						pc.next (+ pc 2)))))

				       ((== ir opc.JNZ)
					;; FIXME: myhdl complains about the if for some reason
					(setf pc.next (+ pc 1))
					#+nil (if (== (aref sr 6) 0)
					    (setf pc.next (+ pc (im.signed)))
					    (setf pc.next (+ pc 1))))
				     #+allop ((== ir opc.JZ)
				       (do0 ;if (== 0 (aref sr 6))
					   (setf pc.next (+ pc 1))
					   ;   (setf pc.next (+ pc (im.signed)))
					   ))
				      ,@(loop for e in `((TAX rx ra rw 1)
							 (TXA ra rx)
							 (INX rx (+ rx 1)
							      rw 1)
							 #+allop (DEX rx (- rx 1)
							      rw 1)
							 #+allop (PHA adr sp
							      sp (- sp 1)
							      di ra
							      we 1)
							 #+allop (PLA sp (+ sp 1)
							      adr (+ sp 1)
							      )
							 (CMP rw 2
							      sr (concat
								  (<= #x80 (- ra im))
								  (== (- ra im) 0)
								  (aref sr (slice 6 0)))
							      pc (+ pc 1))
							 #+allop (JSR adr sp
							      sp (- sp 1)
							      di (>> (+ pc 2) 8)
							      we 1)
							 #+allop (RTS adr (+ sp 1)
							      sp (+ sp 1))
							 #+allop (JMP pc (logior (<< do 8) im))
							 )
					      collect
					      (destructuring-bind (opcode &rest rest) e
						`((== ir (dot opc ,opcode))
						  ,@(loop for (a b) on rest by #'cddr
							  collect
							  `(setf (dot ,a next) ,b)))))
				     #+allop ,@(loop for e in `((ADD +)
							 (SUB -)
							 (AND &)
							 (OR logior)
							 (XOR logxor)
							 (ASL <<)
							 (ASR >>)
							 )
					      collect
					      (destructuring-bind (opcode operator) e
						`((== ir (dot opc ,opcode))
						  (setf ra.next (,operator ra im)
							pc.next (+ pc 1)))))
				      )
				(setf cyc.next M1))

			       ((== cyc M1)
				(cond
				  #+allop ((or (== ir opc.PLA)
				       (and (== ir opc.LDA)
					    (or (== am adm.ABS) ;; fixme precedence?
						(== am adm.IDX))))
				   (setf ra.next do))
				  #+allop ((== ir opc.JSR)
				   (setf adr.next sp
					 sp.next (- sp 1)
					 di.next (& (+ pc 2) #xff)
					 we.next 1
					 pc.next (logior (<< do 8)
							 im))
				   )
				 #+allop ((== ir opc.RTS)
				   (setf pc.next do))
				  (t
				   (setf we.next 0
					 adr.next pc)))
				(setf cyc.next M2))
			       ((== cyc M2)
				(cond ((== ir #x11)
				       (setf ra.next do
					     sr.next (concat (<= #x80 do)
							     (== do 0)
							     (aref sr (slice 6 0)))))
				      ((== rw 0)
				       (setf sr.next (concat (<= #x80 ra)
							     (== ra 0)
							     (aref sr (slice 6 0)))))
				      ((== rw 1)
				       (setf sr.next (concat (<= #x80 ra)
							     (== rx 0)
							     (aref sr (slice 6 0))))))
				;; fixme: is that really a new if?
				(if (== ir #x17)
				    (setf pc.next (logior (<< do 8)
							  (& pc #xff))
					  adr.next (logior (<< do 8)
							   (& pc #xff)))
				    (setf adr.next pc))
				(setf we.next 0
				      rw.next 0
				      cyc.next F1)
				))))
			  (return logic)
			  )))
		   
		   
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
		    (convert_this :hdl (string "Verilog"))))))))
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

