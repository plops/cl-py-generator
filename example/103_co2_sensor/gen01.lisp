(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir*
    ; "/home/martin/src/my_fancy_app_name/main/"
    #P"example/103_co2_sensor/source01/"
    )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))


  (defparameter *full-source-dir*
    ;"/home/martin/src/my_fancy_app_name/main/"
    #-nil
    (asdf:system-relative-pathname
     'cl-py-generator
     *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (write-source
   (asdf:system-relative-pathname
    'cl-py-generator
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     "#define FMT_HEADER_ONLY"
     (include<> deque
		random
		vector
		cmath)

     (include "core.h")


     ,@(let ((n-fifo (* 10 320)))
	 (loop for e in `((N_FIFO ,n-fifo)
			  (RANSAC_MAX_ITERATIONS ,(max n-fifo 12))
			  (RANSAC_INLIER_THRESHOLD 0.1 :type float)
			  (RANSAC_MIN_INLIERS ,(floor (* .1 n-fifo))))
		collect
		(destructuring-bind (name val &key (type 'int)) e
		  (format nil "const ~a ~a = ~a;" type name val))))
     
     
     (defstruct0 Point2D
	 (x double)
       (y double))
     "std::deque<Point2D> fifo(N_FIFO,{0.0,0.0});"


     

     (defun distance (p m b)
       (declare (type Point2D p)
		(type double m b)
		(values double))
       (return (/ (abs (- p.y
			  (+ (* m p.x)
			     b)))
		  (sqrt (+ 1 (* m m))))))


     (defun ransac_line_fit (data m b)
       (declare (type "std::deque<Point2D>&" data)
		(type double& m b))
       (when (< (fifo.size) 2)
	 (return))
       "std::random_device rd;"
       (let (
	     (gen (std--mt19937 (rd)))
	     (distrib (std--uniform_int_distribution<> 0 (- (data.size)
							    1)))
	     (best_inliers (std--vector<Point2D>))
	     (best_m 0d0)
	     (best_b 0d0))
	 (dotimes (i RANSAC_MAX_ITERATIONS)
	   (let ((idx1 (distrib gen))
		   (idx2 (distrib gen))
		   )
	     (while (== idx1 idx2)
		    (setf idx1 (distrib gen)))
	     (let ((p1 (aref data idx1))
		   (p2 (aref data idx2))
		   (m (/ (- p2.y p1.y)
			 (- p2.x p1.x)))
		   (b (- p1.y
			 (* m p1.x)))
		   (inliers (std--vector<Point2D>)))
	       (foreach (p data)
			(when (< (distance p m b)
				 RANSAC_INLIER_THRESHOLD)
			  (inliers.push_back p)))
	       (when (< RANSAC_MIN_INLIERS 
			(inliers.size))
		 (let ((sum_x 0d0)
		       (sum_y 0d0))
		   (foreach (p inliers)
			    (incf sum_x p.x)
			    (incf sum_y p.y))
		   (let ((avg_x (/ sum_x (inliers.size)))
			 (avg_y (/ sum_y (inliers.size)))
			 (var_x 0d0)
			 (cov_xy 0d0))
		     (foreach (p inliers)
			      (incf var_x (* (- p.x avg_x)
					     (- p.x avg_x)))

			      (incf cov_xy (* (- p.x avg_x)
					      (- p.y avg_y))))
		     (let ((m (/ cov_xy var_x))
			   (b (- avg_y (* m avg_x))))
		       (when (< (best_inliers.size)
				(inliers.size))
			 (setf best_inliers inliers
			       best_m m
			       best_b b))))))
	       )))
	 (setf m best_m
	       b best_b)
	 ))
     
     (space
      "extern \"C\" "
      (progn

	(do0
	 (include hardware.h
		  pax_gfx.h
		  pax_codecs.h
		  ili9341.h
		  freertos/FreeRTOS.h
		  freertos/queue.h
		  esp_system.h
		  ;nvs.h
		  ;nvs_flash.h
		  ;wifi_connect.h
		  ;wifi_connection.h
		  soc/rtc.h
		  soc/rtc_cntl_reg.h
		  ;gpio_types.h
		  driver/uart.h
		  sys/time.h)

	 (include<> esp_log.h))
	;; here they define another uart, uart0:
	;; ../components/mch2022-rp2040/rp2040bl.c


	(do0 "static const char*TAG = \"mch2022-co2-app\";"
	     "static pax_buf_t buf;"
	     "xQueueHandle buttonQueue;")

	(defun disp_flush ()
	  (ili9341_write (get_ili9341)
					; buf.buf
			 #-nil
			 ("static_cast<const uint8_t*>" buf.buf)))

	(defun exit_to_launcher ()
	  (REG_WRITE RTC_CNTL_STORE0_REG 0)
	  (esp_restart))

	"#define CO2_UART UART_NUM_1"
	"#define BUF_SIZE UART_FIFO_LEN" ;; 128

	;;../esp-idf/docs/en/api-reference/peripherals/uart.rst
	
	(defun uart_init ()
	  (ESP_LOGE TAG (string "initialize uart"))
	  (when (uart_is_driver_installed CO2_UART)
	    (return))

	  #+nil (do0 (comments "i think uart_set_pin will configure the ports (and also check that they are valid)")
	   ,@(loop for e in `((:gpio 27 :mode GPIO_MODE_OUTPUT)
			      (:gpio 39 :mode GPIO_MODE_INPUT)
			      )
		   collect
		   (destructuring-bind (&key gpio mode) e
		     `(unless (== ESP_OK (gpio_set_direction ,gpio ,mode))
			(ESP_LOGE TAG (string ,(format nil "error initializing gpio ~a" gpio)))))))
	  
	 (unless (== ESP_OK (uart_set_pin CO2_UART
				 27 ;; tx
				 39 ;; rx
				 UART_PIN_NO_CHANGE
				 UART_PIN_NO_CHANGE))

			  (ESP_LOGE TAG (string "error: uart_set_pin 27 39")))
	  (unless (== ESP_OK
		   (uart_driver_install CO2_UART
					200 ;BUF_SIZE ;; rx
					0	 ;BUF_SIZE ;; tx
					0	 ;; queue length
 					nullptr	 ;; queue out
					0	 ;; interrupt
					)
		   )
	    (ESP_LOGE TAG (string "error: uart_driver_install"))
	    )
	  (let (
		(config (uart_config_t
			 (designated-initializer
			  :baud_rate 9600
			  :data_bits UART_DATA_8_BITS
			  :parity UART_PARITY_DISABLE
			  :stop_bits UART_STOP_BITS_1
			  :flow_ctrl UART_HW_FLOWCTRL_DISABLE
			  ;:rx_flow_ctrl_thresh 0
			  :source_clk UART_SCLK_APB))))
	    
	    (unless (== ESP_OK (uart_param_config CO2_UART &config))
	       (ESP_LOGE TAG (string "error: uart_param_config")))
	    ))

	
	(defun measureCO2 ()
	  (progn
	    (ESP_LOGE TAG (string "measure co2"))
	    ,(let ((l `(#xff #x01 #x86 0 0 0 0 0 #x79)))
	       `(let ((command (curly ,@(loop for e in l
					      collect
					      `(hex ,e))))
		      (response))
		  (declare (type (array "unsigned char" ,(length l)) command response))
		  (uart_write_bytes CO2_UART command ,(length l))
		  (let ((l (uart_read_bytes CO2_UART response ,(length l)
					    100)))
		    (when (== 9 l)
		      (when (logand (== #xff (aref response 0))
				    (== #x86 (aref response 1)))
			(let ((co2 (+ (* 256 (aref response 2))
				      (aref response 3))))
			  (ESP_LOGE TAG (string "%s") (dot ,(sprint  :vars `(co2))
						(c_str)))
			  (when (< (- N_FIFO 1) (fifo.size))
			    (fifo.pop_back))
			  (let ((tv_now (timeval )))
			    (gettimeofday &tv_now nullptr)
			   (let ((time_us (+ tv_now.tv_sec (* 1e-6 tv_now.tv_usec)))
				 (p (Point2D (designated-initializer :x time_us
								     :y (static_cast<double> co2)))))
			     (fifo.push_front p)))))))))))

       
	
	(defun scaleHeight (v)
	  (declare (type float v)
		   (values float))
	  (comments "v is in the range 400 .. 5000"
		    "map to 0 .. 239")
	  (let ((mi 400s0)
		(ma 5000s0)
		(res (* 239s0 (- 1s0
				 (/ (- v mi)
				  (- ma mi))))))
	    (when (< res 0s0)
	      (setf res 0s0))
	    (when (< 239s0 res)
	      (setf res 239s0))
	    (return res))
	   ;; note: i think  to and bottom are inverted on screen 
	  )
	
	(defun drawCO2 (buf )
	  (declare (type "pax_buf_t*" buf))
	  (when (< (fifo.size) 2)
	    (return))
	  (let ((hue 12)
		(sat 255)
		(bright 255)
		(col (pax_col_hsv hue
				  sat bright))
		)
	    (let ((time_ma (dot (aref fifo 0) x))
		  (time_mi (dot (aref fifo (- (dot fifo (size)) 1))
				x))
		  (time_delta (- time_ma time_mi))
		  (scaleTime (lambda (x)
			       (declare (type float x)
					(capture "&")
					(values float))
			       (return (* 319s0 (/ (- x time_mi )
						 time_delta))))))
	     (dotimes (i (- (fifo.size) 1))
	       (let ((a (aref fifo i))
		     (b (aref fifo (+ i 1))))
		 (pax_draw_line buf col
				(scaleTime a.x)
				(scaleHeight a.y)
				(scaleTime b.x)
				(scaleHeight b.y))))

	      (progn
		(let ((m 0d0)
		      (b 0d0)
		      (hue 202)
		      (sat 255)
		      (bright 255)
		      (col (pax_col_hsv hue
					sat bright))
		      )
		  (ransac_line_fit fifo
				   m b)
		  (pax_draw_line buf col
				(scaleTime time_mi)
				(scaleHeight (+ b (* m time_mi)))
				(scaleTime time_ma)
				(scaleHeight (+ b (* m time_ma)))
				)
		  )

		(do0
		 (comments "compute when a value of 1200ppm is reached")
		 (let ((x0 (/ (- 1200d0 b)
			      m)))
		   (if (< time_ma x0)
		       (do0 (comments "if predicted intersection time is in the future, print it")
			    (let ((time_value (static_cast<int> (- x0 time_ma)))
				  (hours (int (/ time_value 3600)))
				  (minutes (int (/ (% time_value 3600) 60)))
				  (seconds (% time_value 60))
				  (text_ (fmt--format (string "air room in (h:m:s) {:02d}:{:02d}:{:02d}")
						      hours minutes seconds))
				  (text (text_.c_str))
				  (font pax_font_sky)
				  (dims (pax_text_size font
						       font->default_size
						       text)))
		 	      
			      (pax_draw_text buf
					     
					     (hex #xffffffff) ; white
					     font
					     font->default_size
					     20
					     140
					     text)
			      
			      ))
		       (do0 (comments "if predicted intersection time is in the past, then predict when airing should stop")
			    (let ((x0 (/ (- 500d0 b)
					 m))
				  (time_value (static_cast<int> (- x0 time_ma)))
				  (hours (int (/ time_value 3600)))
				  (minutes (int (/ (% time_value 3600) 60)))
				  (seconds (% time_value 60))
				  (text_ (fmt--format (string "air of room should stop in (h:m:s) {:02d}:{:02d}:{:02d}")
						      hours minutes seconds))
				  (text (text_.c_str))
				  (font pax_font_sky)
				  (dims (pax_text_size font
						       font->default_size
						       text)))
		 	      
			      (pax_draw_text buf
					     
					     (hex #xffffffff) ; white
					     font
					     font->default_size
					     20
					     140
					     text)
			      
			      )))))

		)

	      ))
	  )
	
	(defun app_main ()
	  (ESP_LOGI TAG (string "welcome to the template app"))
	  (bsp_init) 
	  
	  (bsp_rp2040_init)
	  (setf buttonQueue (-> (get_rp2040)
				queue))
	  (pax_buf_init &buf nullptr 320 240 PAX_BUF_16_565RGB)
	  ;(nvs_flash_init)
	  ;(wifi_init)
	  (uart_init)

	  
	  
	  (while 1
		 (measureCO2)
		 (let ((hue 129 #+nil
			    (and (esp_random)
				 255
				 ))
		       (sat 0)
		       (bright 0)
		       (col (pax_col_hsv hue
					 sat bright))
		       )
		   (pax_background &buf col)
		   (let (
			 (text_ ,(sprint :msg (multiple-value-bind
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
			 (text (text_.c_str))
			 (font pax_font_sky ;saira_condensed
			       )
			 (dims (pax_text_size font
					      font->default_size
					      text)))
		     (drawCO2 &buf)

		     
		     
		     (pax_draw_text &buf
					; (hex #xff000000) ; black
				    (hex #xffffffff) ; white
				    font
				    font->default_size
				    (/ (- buf.width
					  dims.x)
				       2.0)
				    (/ (- buf.height
					  dims.y)
				       2.0)
				    text)
		     
		     (progn
		       (let ((now (dot (aref fifo 0) x))
			     (nowtext_ ,(sprint :vars `(now))))
			(pax_draw_text &buf
				       (hex #xffffffff) ; white
				       font
				       font->default_size
				       20
				       180
				       (nowtext_.c_str)))
		       (let (
			     (co2 (dot (aref fifo 0) y))
			     (text_ ,(sprint :vars `(co2)))
			     (font pax_font_saira_condensed)
			     (text (text_.c_str))
			     (dims (pax_text_size font
						  font->default_size
						  text)))
			 
			 (pax_draw_text &buf
					(hex #xffffffff) ; white
				       font
				       font->default_size
				       (/ (- buf.width
					  dims.x)
					  2.0)
				       200
				       text
				       )))
		     
		     (disp_flush)
		     
		     (let ((message (rp2040_input_message_t)))
		       (xQueueReceive buttonQueue
				      &message
				      100 ;portMAX_DELAY
				      )

		       (when (logand (== RP2040_INPUT_BUTTON_HOME
					 message.input)
				     message.state)
			 (exit_to_launcher))))))))))))



