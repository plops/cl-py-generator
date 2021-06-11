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
		      `(setf ,(format nil "~a_~a" dir name) (intbv ,value))))
	    (setf pixel_for_hs (+ h_extent h_back h_front)
		  line_for_vs (+ v_extent v_back v_front))
	    (setf pixel_count (Signal (intbv 0 :min 0 :max (+ h_extent h_back h_front 100)))
		  line_count (Signal (intbv 0 :min 0 :max (+ v_extent v_back v_front 100))))

	    @block
	    (def lcd (pixel_clk n_rst data_r data_g data_b)
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
			   data_b 0
			   data_g 0
			   )
		     
		     )))
	      (return (tuple logic_count logic_data)))


	    
	    
	    (do0
	     
	     (def convert_this (hdl)
	       (do0 
		,@(loop for e in `(pixel_clk n_rst)
			collect
			`(setf ,e (Signal (bool 0))))
		,@(loop for (e f) in `((data_r 5) (data_g 6) (data_b 5)
				       )
			collect
			`(setf ,e (Signal (aref (intbv 0) (slice ,f "")))))
	
		(setf lcd_1 (lcd pixel_clk n_rst data_r data_b data_g))
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
	(loop for (e f) in `((sys_clk 35)
			   (sys_rst_n 15)
			   ("led[0]" 16)
			   ("led[1]" 17)
			   ("led[2]" 18)
			   )
	      do
	      (p (format nil "IO_LOC \"~a\" ~a" e f)))))))



