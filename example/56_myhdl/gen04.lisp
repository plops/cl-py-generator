(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)


(push :color *features*)

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/56_myhdl")
  (defparameter *code-file* "run_04_lcd")
  (defparameter *source* (format nil "~a/source/04_tang_lcd/" *path*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (let* ((code
	   `(do0
	     (imports-from (myhdl *)
			   (random randrange))
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
	     (comments "https://tangnano.sipeed.com/en/examples/2_lcd.html")
	     (comments "https://github.com/sipeed/Tang-Nano-examples/blob/master/example_lcd/lcd_pjt/src/VGAMod.v")
	     (comments "AT050TN43.pdf ILI6122.pdf ")
	     ,@(loop for e in `((v back 6)
				(v pulse 5)
				(v extent 272 ; 480
				   )
				(v front  62
				   ) ;; 45
				(h back 182)
				(h pulse 1)
				(h extent 480 ;800
				   )
				(h front 210))
		     collect
		     (destructuring-bind (dir name value) e
		       `(setf ,(format nil "~a_~a" dir name) ,value)))
	     (setf pixel_for_hs (+ h_extent h_back h_front)
		   line_for_vs (+ v_extent v_back v_front))
	     (setf pixel_count (Signal (intbv 0 :min 0 :max (+ h_extent h_back h_front 100)))
		   line_count (Signal (intbv 0 :min 0 :max (* 2  (+ v_extent v_back v_front 100)))))
	     ,@ (loop for e in `(r g b)
		      collect
		      `(setf ,(format nil "data_~a" e)
			     (Signal (aref (intbv 0) (slice 10 "")))))
	     	(do0
		 @block
	      (def lcd (pixel_clk n_rst lcd_de lcd_hsync lcd_vsync lcd_r lcd_g lcd_b)
		(do0
		 (@always pixel_clk.posedge n_rst.negedge)
		 (def logic_count ()
		   (if (== n_rst 0)
		       (setf line_count.next 0
			     pixel_count.next 0)
		       (if (== pixel_count pixel_for_hs)
			   (setf line_count.next (+ line_count 1)
				 pixel_count.next 0)
			   (if (== line_count line_for_vs)
			       (setf line_count.next 0
				     pixel_count.next 0)
			       (setf pixel_count.next (+ pixel_count 1)))))))
		#+nil(do0
		 (@always pixel_clk.posedge n_rst.negedge)
		 (def logic_data ()
		   (when (== n_rst 0)
		     (setf data_r.next 0
			   data_b.next 0
			   data_g.next 0))))
		(do0
		 @always_comb
		 (def logic_sync ()
		   #+nil (setf lcd_hsync
			       (? (& (<= h_pulse
					 pixel_count )
				     (<= pixel_count (+ h_extent h_back) ; (- pixel_for_hs h_front) 
					 ))
				  0 1))
		   #-nil(if (& (<= h_pulse
			      pixel_count )
			  (<= pixel_count (+ h_extent h_back) ; (- pixel_for_hs h_front) 
			      ))
		       (setf lcd_hsync.next 0)
		       (setf lcd_hsync.next 1))
		   (if (& (<= v_pulse
			      line_count)
			  (<= line_count 
			      line_for_vs))
		       (setf lcd_vsync.next 0)
		       (setf lcd_vsync.next 1))
		   (if (& (& (<= h_back pixel_count)
			     (<= pixel_count (+ h_extent h_back)))
			  (& (<= v_back
				 line_count)
			     (<= line_count 
				 (+ v_extent 5 ;(- v_back 1)
				    ))))
		       (setf lcd_de.next 1)
		       (setf lcd_de.next 0))))
		(do0
		 @always_comb
		 (def logic_pattern ()
		   #-color
		   (cond ((< pixel_count 240)
			  (setf lcd_r.next 1))
			 ((< pixel_count 480)
			  (setf lcd_g.next 1))
			 ((< pixel_count 600)
			  (setf lcd_b.next 1))
			 (t
			  (setf lcd_r.next 0
				lcd_g.next 0
				lcd_b.next 0))
			 )
		   #+color ,@(loop for (e f) in `((lcd_r 400)
					  (lcd_b 640)
					  (lcd_g 840))
			   collect
			   `(do0
			     ,(let ((res `(setf (dot ,e next) 0)))
				(loop for val in `(16 8 4 2 1 0) and i from 0
				      do
					 (setf res `(if (< pixel_count ,(- f (* 40 i)))
							(setf (dot ,e next) ,val)
							,res)))
				res)))))
		(return (tuple logic_count ;logic_data
			       logic_sync logic_pattern))))
	     (do0
	      (def convert_this (hdl)
		(do0 
		 #-nil
		 (do0
		  (do0 ,@(loop for e in `(pixel_clk ;n_rst
				      lcd_de lcd_hsync 
				      lcd_vsync
				      #-color lcd_r
				      #-color lcd_g
				      #-color lcd_b
				      )
			   collect
			       `(setf ,e (Signal (bool 0))))
		       (setf n_rst (ResetSignal 0 :active 0 :isasync False))
		       #+color ,@(loop for (e f) in `((lcd_r 5) (lcd_g 6) (lcd_b 5)
					      )
			       collect
				       `(setf ,e (Signal (aref (intbv 0) (slice ,f "")))))
		       )
		     
		     
		     (do0 (setf lcd_1 (lcd pixel_clk n_rst
					   lcd_de lcd_hsync lcd_vsync
					   lcd_r lcd_g lcd_b
					   ))
			  (lcd_1.convert :hdl hdl)))))
	      (convert_this :hdl (string "Verilog"))))))
    (write-source (format nil "~a/~a" *source* *code-file*) code)
    (with-open-file (s (format nil "~a/~a" *source* "rpll.v")
		       :direction :output
		       :if-exists :supersede)
      (format s "~a"
	      "module Gowin_rPLL (clkout, clkoutd, clkin);

output clkout;
output clkoutd;
input clkin;

wire lock_o;
wire clkoutp_o;
wire clkoutd3_o;
wire gw_gnd;

assign gw_gnd = 1'b0;

rPLL rpll_inst (
    .CLKOUT(clkout),
    .LOCK(lock_o),
    .CLKOUTP(clkoutp_o),
    .CLKOUTD(clkoutd),
    .CLKOUTD3(clkoutd3_o),
    .RESET(gw_gnd),
    .RESET_P(gw_gnd),
    .CLKIN(clkin),
    .CLKFB(gw_gnd),
    .FBDSEL({gw_gnd,gw_gnd,gw_gnd,gw_gnd,gw_gnd,gw_gnd}),
    .IDSEL({gw_gnd,gw_gnd,gw_gnd,gw_gnd,gw_gnd,gw_gnd}),
    .ODSEL({gw_gnd,gw_gnd,gw_gnd,gw_gnd,gw_gnd,gw_gnd}),
    .PSDA({gw_gnd,gw_gnd,gw_gnd,gw_gnd}),
    .DUTYDA({gw_gnd,gw_gnd,gw_gnd,gw_gnd}),
    .FDLY({gw_gnd,gw_gnd,gw_gnd,gw_gnd})
);

defparam rpll_inst.FCLKIN = \"24\";
defparam rpll_inst.DYN_IDIV_SEL = \"false\";
defparam rpll_inst.IDIV_SEL = 2;
defparam rpll_inst.DYN_FBDIV_SEL = \"false\";
defparam rpll_inst.FBDIV_SEL = 24;
defparam rpll_inst.DYN_ODIV_SEL = \"false\";
defparam rpll_inst.ODIV_SEL = 4;
defparam rpll_inst.PSDA_SEL = \"0000\";
defparam rpll_inst.DYN_DA_EN = \"true\";
defparam rpll_inst.DUTYDA_SEL = \"1000\";
defparam rpll_inst.CLKOUT_FT_DIR = 1'b1;
defparam rpll_inst.CLKOUTP_FT_DIR = 1'b1;
defparam rpll_inst.CLKOUT_DLY_STEP = 0;
defparam rpll_inst.CLKOUTP_DLY_STEP = 0;
defparam rpll_inst.CLKFB_SEL = \"internal\";
defparam rpll_inst.CLKOUT_BYPASS = \"false\";
defparam rpll_inst.CLKOUTP_BYPASS = \"false\";
defparam rpll_inst.CLKOUTD_BYPASS = \"false\";
defparam rpll_inst.DYN_SDIV_SEL = 6;
defparam rpll_inst.CLKOUTD_SRC = \"CLKOUT\";
defparam rpll_inst.CLKOUTD3_SRC = \"CLKOUT\";
defparam rpll_inst.DEVICE = \"GW1N-1\";
endmodule"))
     (with-open-file (s (format nil "~a/~a" *source* "lcd_project.gprj")
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
	      (loop for (e f) in `((top.v verilog)
				   (lcd.v verilog)
				   ;(osc.v verilog)
				   (rpll.v verilog)
				   (test.cst cst))
		    collect
		    (format nil "<File path=\"~a\" type=\"file.~a\" enable=\"1\"/>" e f))))

  #+nil  (with-open-file (s (format nil "~a/~a" *source* "osc.v")
		       :direction :output
		       :if-exists :supersede)
      (format s "~a"
	      "module Gowin_OSC (oscout);

output oscout;

OSCH osc_inst (
    .OSCOUT(oscout)
);

defparam osc_inst.FREQ_DIV = 10;

endmodule"))
    (with-open-file (s (format nil "~a/~a" *source* "top.v")
		       :direction :output
		       :if-exists :supersede)
      (labels ((p (rest)
		 (format s "~{~a~%~}" rest))
	       (paren (rest)
		 
		 (format s "(~{~a~^,~})" rest)))
	(p `("module top"))
	(paren
	 (append (loop for e in `(n_rst xtal_in)
		       collect
		       (format nil "input ~a" e))
		 (loop for e in `(lcd_clk lcd_hsync lcd_vsync lcd_de
					  #+color "[4:0] lcd_r"
					  #+color "[5:0] lcd_g"
					  #+color "[4:0] lcd_b"

					  #-color lcd_r
					  #-color lcd_g
					  #-color lcd_b
					; led_r led_g led_b
					;key
					  )
		       collect
		       (format nil "output ~a" e))))
	(p `(";"))
	(p 
	   (loop for e in `(clk_sys clk_pix; oscout_o
				      )
		       collect
		       (format nil "wire ~a;" e))
	   )
	(p `("Gowin_rPLL chip_pll"))
	(paren (loop for  (e f) in `((clkout clk_sys)
				       (clkoutd clk_pix)
				       (clkin xtal_in))
		       collect
		       (format nil ".~a(~a)" e f)))
	(p `(";"))
	(p `("lcd lcd_1"))
	(paren (loop for  e in `(;(clk clk_sys)
				   n_rst 
				   (pixel_clk clk_pix)
				   lcd_de 
				   lcd_hsync
				   lcd_vsync
				   lcd_r
				   lcd_g
				   lcd_b)
		     collect
		     (if (listp e)
			 (format nil ".~a(~a)" (first e) (second e))
			 (format nil ".~a(~a)" e e )))
	       )
	(p `(";"))
	(p `("assign lcd_clk = clk_pix;"))
	(p `("endmodule"))))
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
	#+color
	(loop for e in `((r (0 4) (27 28 29 30 31))
			 (g (0 5) (32 33 34 38 39 40))
			 (b (0 4) (41 42 43 44 45)))
	      
	      do
		 (destructuring-bind (name (start end) vals) e
		   (loop for v in vals and i from start upto end
			 do
			    (p (format nil "IO_LOC \"lcd_~a[~a]\" ~a" name i v)))))

	
	#-color (loop for e in `((r 31)
			 (g 40)
			 (b 45))
		      ;; only most significant color bit
	      do
		 (destructuring-bind (name v) e
		   (p (format nil "IO_LOC \"lcd_~a\" ~a" name v))))
	(loop for e in `(n_rst lcd_clk lcd_vsync lcd_de ;key
			       xtal_in lcd_hsync
			       #+color (lcd_r 4) #+color (lcd_g 5) #+color (lcd_b 4)
			       #-color lcd_r #-color lcd_g #-color lcd_b
			       )
	      do
		 (if (listp e)
		     (destructuring-bind (name i) e
		       (loop for count from 0 upto i do
			(p (format nil "IO_PORT \"~a[~a]\" IO_TYPE=LVCMOS33" name count))))
		     (p (format nil "IO_PORT \"~a\" IO_TYPE=LVCMOS33" e))))))))

