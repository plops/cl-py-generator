(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



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

	     (comments "https://tangnano.sipeed.com/en/examples/2_lcd.html")
	     (comments "https://github.com/sipeed/Tang-Nano-examples/blob/master/example_lcd/lcd_pjt/src/VGAMod.v")
	     (comments "AT050TN43.pdf ILI6122.pdf ")

	     ,@(loop for e in `((v back 6)
				(v pulse 5)
				(v extent 480)
				(v front 62) ;; 45
				(h back 182)
				(h pulse 1)
				(h extent 800)
				(h front 210))
		     collect
		     (destructuring-bind (dir name value) e
		       `(setf ,(format nil "~a_~a" dir name) ,value)))
	     (setf pixel_for_hs (+ h_extent h_back h_front)
		   line_for_vs (+ v_extent v_back v_front))
	     (setf pixel_count (Signal (intbv 0 :min 0 :max (+ h_extent h_back h_front 100)))
		   line_count (Signal (intbv 0 :min 0 :max (+ v_extent v_back v_front 100))))

	     ,@ (loop for e in `(r g b)
		      collect
		      `(setf ,(format nil "data_~a" e)
			     (Signal (aref (intbv 0) (slice 10 "")))))
	     
		(do0
		 @block
		 (def TOP (n_rst
			   xtal_in
			   lcd_clk
			   lcd_hsync
			   lcd_vsync
			   lcd_de
			   lcd_r
			   lcd_g
			   lcd_b
			  ; key
			   )
		   
		   (do0
		    @always_comb
		    (def logic ()
		      (setf lcd_clk clk_pix)))
		   
		   (setf lcd_1 (lcd clk_pix n_rst  lcd_de lcd_hsync lcd_vsync lcd_r lcd_g lcd_b))
		   (return logic))
		 )
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
			       (setf pixel_count.next (+ pixel_count 1))))
		       )))
		(do0
		 (@always pixel_clk.posedge n_rst.negedge)
		 (def logic_data ()
		   (when (== n_rst 0)
		     (setf data_r.next 0
			   data_b.next 0
			   data_g.next 0
			   )
		     
		     )))
		(do0
		 @always_comb
		 (def logic_sync ()
		   #+nil (setf lcd_hsync
			       (? (& (<= h_pulse
					 pixel_count )
				     (<= pixel_count (+ h_extent h_back) ; (- pixel_for_hs h_front) 
					 
					 
					 ))
				  0 1))
		   (if (& (<= h_pulse
			      pixel_count )
			  (<= pixel_count (+ h_extent h_back) ; (- pixel_for_hs h_front) 
			      
			      
			      ))
		       (setf lcd_hsync 0)
		       (setf lcd_hsync 1)
		       
		       )
		   (if (& (<= v_pulse
			      line_count)
			  (<= line_count 
			      line_for_vs))
		       (setf lcd_vsync 0)
		       (setf lcd_vsync 1))
		   (if (& (& (<= h_back pixel_count)
			     (<= pixel_count (+ h_extent h_back)))
			  (& (<= v_back
				 line_count)
			     (<= line_count 
				 (+ v_extent 5 ;(- v_back 1)
				    )))
			  )
		       (setf lcd_de 1
			     )
		       (setf lcd_de 0))))

		(do0
		 @always_comb
		 (def logic_pattern ()
		   ,@(loop for (e f) in `((lcd_r 400)
					  (lcd_b 640)
					  (lcd_g 840)
					  )
			   collect
			   `(do0
			     ,(let ((res `(setf (dot ,e next) 0)))
				(loop for val in `(16 8 4 2 1 0) and i from 0
				      do
					 (setf res `(if (< pixel_count ,(- f (* 40 i)))
							(setf (dot ,e next) ,val)
							,res)))
				res)))))
		(return (tuple logic_count logic_data logic_sync logic_pattern))))


	     
	     
	     (do0
	      
	      (def convert_this (hdl)
		(do0 
		 ,@(loop for e in `(pixel_clk ;n_rst
				    lcd_de lcd_hsync 
				    lcd_vsync)
			 collect
			 `(setf ,e (Signal (bool 0))))
		 (setf n_rst (ResetSignal 0 :active 0 :isasync False))
		 ,@(loop for (e f) in `((lcd_r 5) (lcd_g 6) (lcd_b 5)
					)
			 collect
			 `(setf ,e (Signal (aref (intbv 0) (slice ,f "")))))
		 
		 (setf lcd_1 (lcd pixel_clk n_rst
				  lcd_de lcd_hsync lcd_vsync
				  lcd_r lcd_b lcd_g
				  ))
		 (lcd_1.convert :hdl hdl))
		)
	      (convert_this :hdl (string "Verilog")))
	     )))
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
	      (loop for (e f) in `((lcd.v verilog)
				   (osc.v verilog)
				   (rpll.v verilog)
				   (test.cst cst))
		    collect
		    (format nil "<File path=\"~a\" type=\"file.~a\" enable=\"1\"/>" e f))))

    (with-open-file (s (format nil "~a/~a" *source* "osc.v")
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
    (with-open-file (s (format nil "~a/~a" *source* "test.cst")
		       :direction :output
		       :if-exists :supersede :if-does-not-exist :create)
      (flet ((p (str)
	       (format s "~a;~%" str)))
	(loop for (e f) in `((xtal_in 35)
			     (n_rst 14)
			     (key 15)
			     (led_r 16)
			     (led_g 17)
			     (led_b 18)
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
			 (g (0 5) (32 33 34 38 39))
			 (b (0 4) (41 42 43 44 45))
			 )
	      
	      do
		 (destructuring-bind (name (start end) vals) e
		   (loop for v in vals and i from start upto end
			 do
			    (p (format nil "IO_LOC \"lcd_~a[~a]\" ~a" name i v)))))
	(loop for e in `(n_rst lcd_clk lcd_vsync lcd_de key xtal_in lcd_hsync
			       (lcd_r 4) (lcd_g 5) (lcd_b 4)
			       )
	      do
		 (if (listp e)
		     (destructuring-bind (name i) e
		       (loop for count from 0 upto i do
			(p (format nil "IO_PORT \"~a[~a]\" IO_TYPE=LVCMOS33" name count))))
		     (p (format nil "IO_PORT \"~a\" IO_TYPE=LVCMOS33" e))))))))

