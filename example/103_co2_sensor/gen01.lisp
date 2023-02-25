(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir*
    "/home/martin/src/my_fancy_app_name/main/"
    ;#P"example/103_co2_sensor/source01/"
    )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))


  (defparameter *full-source-dir*
    "/home/martin/src/my_fancy_app_name/main/"
    #+nil (asdf:system-relative-pathname
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

     

     ,@(loop for e in `((N_FIFO 320)
			    (RANSAC_MAX_ITERATIONS 100)
			    (RANSAC_INLIER_THRESHOLD 0.1 :type float)
			    (RANSAC_MIN_INLIERS 50))
	     collect
	     (destructuring-bind (name val &key (type 'int)) e
	       (format nil "const ~a ~a = ~a;" type name val)))
     
     "std::deque<unsigned short> fifo(N_FIFO,0);"


     (defstruct0 Point2D
       (x double)
       (y double))

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
		 (p1 (aref data idx1))
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
	     ))
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
		  driver/uart.h		  )

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
	"#define BUF_SIZE 64"

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
	  
	  (uart_set_pin CO2_UART
			27 ;; tx
			39 ;; rx
			UART_PIN_NO_CHANGE
			UART_PIN_NO_CHANGE)
	  (uart_driver_install CO2_UART
			       BUF_SIZE ;; rx
			       0 ;BUF_SIZE ;; tx
			       0 nullptr 0)
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
	    
	    (ESP_ERROR_CHECK (uart_param_config CO2_UART &config))
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
			  (when (< N_FIFO (fifo.size))
			    (fifo.pop_back))
			  (fifo.push_front co2)))))))))

	(defun scaleHeight (v)
	  (declare (type float v)
		   (values float))
	  (comments "v is in the range 400 .. 5000"
		    "map to 0 .. 239")
	  (let ((mi 400s0)
		(ma 5000s0)
		(res (* 239s0 (/ (- v mi)
				 (- ma mi)))))
	    (when (< res 0s0)
	      (setf res 0s0))
	    (when (< 239s0 res)
	      (setf res 239s0))
	    (return res))
	  
	  )
	
	(defun drawCO2 (buf )
	  (declare (type "pax_buf_t*" buf))
	  (let ((hue 12)
		(sat 255)
		(bright 255)
		(col (pax_col_hsv hue
				  sat bright))
		)
	    
	    (dotimes (i (- (fifo.size) 1))
	      
	      (pax_draw_line buf col i
			     (scaleHeight (aref fifo i))
			     (+ i 1)
			     (scaleHeight (aref fifo (+ i 1))))))
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
		       (sat 255)
		       (bright 255)
		       (col (pax_col_hsv hue
					 sat bright))
		       )
		   (pax_background &buf col)
		   (let (;(val (aref fifo (- (fifo.size) 1)))
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
				    (hex #xff000000)
				    font
				    font->default_size
				    (/ (- buf.width
					  dims.x)
				       2.0)
				    (/ (- buf.height
					  dims.y)
				       2.0)
				    text)
		     (disp_flush)
		     
		     (let ((message (rp2040_input_message_t)))
		       (xQueueReceive buttonQueue
				      &message
				      20 ;portMAX_DELAY
				      )

		       (when (logand (== RP2040_INPUT_BUTTON_HOME
					 message.input)
				     message.state)
			 (exit_to_launcher))))))))))))



