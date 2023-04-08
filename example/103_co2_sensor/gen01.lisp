(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

;; features:
;; nowifi .. do not generate code for wifi

(progn
  #+nil
  (progn
    (defparameter *source-dir*       "/home/martin/src/my_fancy_app_name/main/")
    (defparameter *full-source-dir*  "/home/martin/src/my_fancy_app_name/main/"))
  #-nil
  (progn
    (defparameter *source-dir* #P"example/103_co2_sensor/source01/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-py-generator
				     *source-dir*)))
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))


  
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (let ((l-data
	  `((:name temperature :hue 150 ; 
		   
		   :short-name T :unit "°C" :fmt "{:2.2f}")
	    (:name humidity :hue 80	; green
	     :short-name H :unit "%" :fmt "{:2.1f}"
	     )
	    (:name pressure :hue 240	;red
	     :short-name p :unit "mbar" :scale 1s-2 :fmt "{:4.2f}"
	     )
					;(:name gas_resistance :hue 100)
	    )))
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
		  algorithm
		  cmath)

       (include "core.h")


       ,@(let ((n-fifo (floor 320 1)))
	   (loop for e in `((N_FIFO ,n-fifo)
			    (RANSAC_MAX_ITERATIONS ,(max n-fifo 12))
			    (RANSAC_INLIER_THRESHOLD 5.0 :type float)
			    (RANSAC_MIN_INLIERS 2 ;,(floor (* .03 n-fifo))
						))
		 collect
		 (destructuring-bind (name val &key (type 'int)) e
		   (format nil "const ~a ~a = ~a;" type name val))))
     
     
       (defstruct0 Point2D
	   (x double)
	 (y double))
       (defstruct0 PointBME
	   (x double)
	 (temperature double)
	 (humidity double)
	 (pressure double)
					;(gas_resistance double)
	 )
       "std::deque<Point2D> fifo; "	;(N_FIFO,{0.0,0.0});
       "std::deque<PointBME> fifoBME; " ;(N_FIFO,{0.0,0.0,0.0,0.0});


     (do0
	   #-nowifi
	   (do0
	    (space "extern \"C\" "
		   (progn
		     (include "esp_wifi.h"
			      ;"esp_netif_types.h"
			      "nvs_flash.h"
			      "freertos/FreeRTOS.h"
			      "freertos/task.h"
			      "freertos/event_groups.h"
			      )
		     (include lwip/sockets.h)
		     (include<> arpa/inet.h
				) ;; inet_ntoa
		     (include "secret.h")
		     (space static EventGroupHandle_t s_wifi_event_group)
		     (comments "event group should allow two different events"
			       "1) we are connected to access point with an ip"
			       "2) we failed to connect after a maximum amount of retries")
		     
		     ;(space static const char* (setf TAG (string "wifi station")))
		     (space static int (setf s_retry_num 0))))
	    
	    (space enum "{"
		   (comma
		    (= WIFI_CONNECTED_BIT BIT0)
		    (= WIFI_FAIL_BIT BIT1)
		    (= EXAMPLE_ESP_MAXIMUM_RETRY 7))
		   "}")
	    (defun event_handler (arg
				  event_base
				  event_id
				  event_data)
	      (declare (type void* arg event_data)
		       (type esp_event_base_t event_base)
		       (type int32_t event_id)
		       (static))
	      (if (logand (== WIFI_EVENT
			      event_base)
			  (== WIFI_EVENT_STA_START
			      event_id))
		  (esp_wifi_connect)
		  (if (logand (== WIFI_EVENT event_base)
			      (== WIFI_EVENT_STA_DISCONNECTED event_id))
		      (do0
		       (if (< s_retry_num EXAMPLE_ESP_MAXIMUM_RETRY)
			   (do0
			    (esp_wifi_connect)
			    (incf s_retry_num)
			    ,(lprint :msg
				      "retry to connect to the access point"))
			   (xEventGroupSetBits s_wifi_event_group
					       WIFI_FAIL_BIT))
		       ,(lprint :msg "connection to the access point failed"))
		      (when (logand (== IP_EVENT event_base)
				    (== IP_EVENT_STA_GOT_IP event_id))
			(let ((event (static_cast<ip_event_got_ip_t*> event_data)))

			  ,(lprint :msg "got ip:"
				   :vars `(
					   (IP2STR &event->ip_info.ip)))
			  (setf s_retry_num 0)
			  (xEventGroupSetBits
			   s_wifi_event_group
			   WIFI_CONNECTED_BIT))))))
	    (defun wifi_init_sta ()
	      (setf s_wifi_event_group (xEventGroupCreate))
	      (ESP_ERROR_CHECK (esp_netif_init))
	      (ESP_ERROR_CHECK (esp_event_loop_create_default))
	      (esp_netif_create_default_wifi_sta)
	      (let ((cfg (space wifi_init_config_t (WIFI_INIT_CONFIG_DEFAULT))))
		(ESP_ERROR_CHECK (esp_wifi_init &cfg))
		(let ((instance_any_id (esp_event_handler_instance_t))
		      (instance_got_ip (esp_event_handler_instance_t)))
		  (ESP_ERROR_CHECK
		   (esp_event_handler_instance_register
		    WIFI_EVENT
		    ESP_EVENT_ANY_ID
		    &event_handler
		    nullptr
		    &instance_any_id))
		  (ESP_ERROR_CHECK
		   (esp_event_handler_instance_register
		    IP_EVENT
		    IP_EVENT_STA_GOT_IP
		    &event_handler
		    nullptr
		    &instance_got_ip))
		  (let ((wifi_config "{}"))
		    (declare (type wifi_config_t wifi_config))
		    ,@(loop for e in `((:key ssid :value (string "mi") :copy t)
				       (:key password :value WIFI_SECRET :copy t)
				       (:key threshold.authmode :value WIFI_AUTH_WPA2_PSK)
				       (:key pmf_cfg.capable :value true)
				       (:key pmf_cfg.required :value false))
			    collect
			    (destructuring-bind (&key key value copy) e
			     (if copy
				 (let ((str (format nil "~a_str" key)))
				  
				   `(let ((,str ,value))
				      (declare (type "const char*" ,str))
				      (std--memcpy  (dot wifi_config sta ,key)
						    ,str
						    (std--strlen ,str))))
				 `(setf (dot wifi_config sta ,key)
					,value)))))
		  (ESP_ERROR_CHECK (esp_wifi_set_mode WIFI_MODE_STA))
		  (ESP_ERROR_CHECK (esp_wifi_set_config WIFI_IF_STA &wifi_config))
		  (ESP_ERROR_CHECK (esp_wifi_start))
		  ;(ESP_LOGI TAG (string "wifi_init_sta finished"))
		  ,(lprint :msg "wait until connection is established or connection failed s_retry_num times"
			   :vars `(s_retry_num))
		  (let ((bits (xEventGroupWaitBits
			       s_wifi_event_group
			       (or WIFI_CONNECTED_BIT
				   WIFI_FAIL_BIT)
			       pdFALSE
			       pdFALSE
			       portMAX_DELAY)))
		    (if (and WIFI_CONNECTED_BIT
			     bits)
			,(lprint :msg "connected to ap")
			(if (and WIFI_FAIL_BIT
			     bits)
			    ,(lprint :msg "connection to ap failed")
			    ,(lprint :msg "unexpected event")))

		    )
		  (ESP_ERROR_CHECK (esp_event_handler_instance_unregister
				    IP_EVENT
				    IP_EVENT_STA_GOT_IP
				    instance_got_ip
				    ))
		  (ESP_ERROR_CHECK (esp_event_handler_instance_unregister
				    WIFI_EVENT
				    ESP_EVENT_ANY_ID
				    instance_any_id
				    ))
		  (vEventGroupDelete s_wifi_event_group))
		))

	    (defun connect_to_tcp_server ()
	      (declare (values esp_err_t))
	      (let ((port 12345)
		    (ip_address (string "192.168.120.122"))
		    (addr ((lambda ()
			     (declare (constexpr)
				      (capture port ip_address)
				      (values sockaddr_in))
			     "sockaddr_in addr{};"
			     (setf addr.sin_family AF_INET
				   addr.sin_port (htons port)
				   )
			     (inet_pton AF_INET ip_address &addr.sin_addr)
			     (return addr))))
		    (domain AF_INET)
		    (type SOCK_STREAM)
		    (protocol 0)
		    (sock (socket domain type protocol)))
		(when (< sock 0)
		  ,(lprint :msg "failed to create socket")
		  (return -1))
		(when (!= 0 (connect sock
				     ("reinterpret_cast<const sockaddr*>" &addr)
				     (sizeof addr))) 
		  ,(lprint :msg "failed to connect to socket"
			   )
		  (close sock)
		  (return -2))
		,(lprint :msg "connected to tcp server")
		(let ((buffer_size 1024)
		      (read_buffer "std::array<char,buffer_size>{}")
		      )
		  (declare (type "constexpr auto" buffer_size))
		  (let ((r (read sock
				 (read_buffer.data)
				 (- (read_buffer.size)
				    1))))
		    (when (< r 0)
		      ,(lprint :msg "failed to read data from socket")
		      (close sock)
		      (return -3))
		    (setf (aref read_buffer r)
			  (char "\\0"))
		    ,(lprint :msg "received data from server"
			     :vars `((read_buffer.data)))))
		(return 0)))))

       (defun distance (p m b)
	 (declare (type Point2D p)
		  (type double m b)
		  (values double))
	 (return (/ (abs (- p.y
			    (+ (* m p.x)
			       b)))
		    (sqrt (+ 1 (* m m))))))


       (defun ransac_line_fit (data m b inliers)
	 (declare (type "std::deque<Point2D>&" data)
		  (type "std::vector<Point2D>&" inliers)
		  (type double& m b))
	 (when (< (data.size) 2)
	   (return))
	 "std::random_device rd;"
	 (comments "distrib0 must be one of the 5 most recent datapoints. i am not interested in fit's of the older data")
	 (let (
	       (gen (std--mt19937 (rd)))
	       (distrib0 (std--uniform_int_distribution<> 0 5))
	       (distrib (std--uniform_int_distribution<> 0 (- (data.size)
							      1)))
	       (best_inliers (std--vector<Point2D>))
	       (best_m 0d0)
	       (best_b 0d0))
	   (dotimes (i RANSAC_MAX_ITERATIONS)
	     (let ((idx1 (distrib gen))
		   (idx2 (distrib0 gen))
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
		 b best_b
		 inliers best_inliers)
	   ))
     
       (space
	"extern \"C\" "
	(progn

	  (do0
	   (include hardware.h
		    pax_gfx.h
		    pax_codecs.h
					;  ili9341.h
					; freertos/FreeRTOS.h
					; freertos/queue.h
		    esp_system.h
					;nvs.h
					;nvs_flash.h
					;wifi_connect.h
					;wifi_connection.h
					; soc/rtc.h
		    soc/rtc_cntl_reg.h
		    bme680.h
					;gpio_types.h
		    driver/uart.h
		    sys/time.h)

	   (include<> esp_log.h)

	   )
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
					     200     ;BUF_SIZE ;; rx
					     0	     ;BUF_SIZE ;; tx
					     0	     ;; queue length
 					     nullptr ;; queue out
					     0	     ;; interrupt
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

	  (defun measureBME ()
	    (progn
	      (ESP_LOGE TAG (string "measure BME"))
	      (let ((temperature 0d0)
		    (humidity 0d0)
		    (pressure 0d0)
					;(gas_resistance 0d0)
		    (bme (get_bme680))
		    (s (bme680_status_t)))
		(bme680_set_mode (get_bme680) BME680_MEAS_FORCED)
		(bme680_get_status bme &s)
		(bme680_get_temperature bme 
					&temperature)
		(bme680_get_humidity bme
				     &humidity
				     temperature)
		(bme680_get_pressure bme
				     &pressure
				     temperature)
		#+nil (bme680_get_gas_resistance bme
						 &gas_resistance
						 temperature))

	      (ESP_LOGE TAG (string "%s") (dot ,(sprint  :vars `((== bme nullptr) temperature humidity pressure ;gas_resistance
								 s.new_data
								 s.gas_measuring
								 s.measuring
								 s.gas_measuring_index
								 s.gas_valid
								 s.heater_stable))
					       (c_str)))
	    
	      (when (< (- N_FIFO 1) (fifo.size))
		(fifoBME.pop_back))
	      (let ((tv_now (timeval )))
		(gettimeofday &tv_now nullptr)
		(let ((time_us (+ tv_now.tv_sec (* 1e-6 tv_now.tv_usec)))
		      (p (PointBME (designated-initializer :x time_us
							   :temperature temperature
							   :humidity humidity
							   :pressure pressure
					;:gas_resistance gas_resistance
							   ))))
		  (fifoBME.push_front p)))))
	
	
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

       
	  #+Nil
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

	  ,@(let* ((pitch-y (floor 240 4))
		   (graph-xmax 318s0)
		   (graph-ymin0 1s0)
		   (graph-ymax0 (- pitch-y 1))) 
	      (loop for e in l-data 
		    and e-i from 1
		    collect
		    (let ((graph-ymin (+ graph-ymin0 (* e-i pitch-y)))
			  (graph-ymax (+ graph-ymax0 (* e-i pitch-y))))
		      (destructuring-bind (&key name hue short-name unit fmt (scale 1s0)) e
			`(defun ,(format nil "drawBME_~a" name)
			     (buf)
			   (declare (type "pax_buf_t*" buf))
			   (let ((time_ma (dot (aref fifoBME 0) x))
				 (time_mi (dot (aref fifoBME (- (dot fifoBME (size)) 1))
					       x))
				 (time_delta (- time_ma time_mi))
				 (scaleTime (lambda (x)
					      (declare (type float x)
						       (capture "&")
						       (values float))
					      (let ((res (* ,graph-xmax (/ (- x time_mi )
									   time_delta))))
						(do0 (when (< res 1s0)
						       (setf res 1s0))
						     (when (< ,graph-xmax res)
						       (setf res ,graph-xmax)))
						(return res))))
				 (min_max_y (std--minmax_element (fifoBME.begin)
								 (fifoBME.end)
								 (lambda (p1 p2)
								   (declare (type "const PointBME&" p1 p2))
								   (return (< (dot p1 ,name) (dot p2 ,name)))
								   )))
				 (min_y (-> min_max_y.first ,name))
				 (max_y (-> min_max_y.second ,name))
				 (scaleHeight (lambda (v)
						(declare (type float v)
							 (capture "&")
							 (values float))
				       
						(let ((mi min_y)
						      (ma max_y)
						      (res (+ ,graph-ymin
							      (* ,graph-ymax0 (- 1s0
										 (/ (- v mi)
										    (- ma mi)))))))
						  (do0 (when (< res ,graph-ymin)
							 (setf res ,graph-ymin))
						       (when (< ,graph-ymax res)
							 (setf res ,graph-ymax)))
						  (return res)))))
			     (do0
			      (comments "write latest measurement")
			      (let (
			      
				    (,name (dot (aref fifoBME 0) ,name))
			     
				    (text_ (fmt--format
					    (string ;"co2={:4.0f} T={:2.1f} H={:2.1f}% p={:4.2f} R={:3.3f}"
					     ,(format nil "~a=~a~a" short-name fmt unit)
					     )
					    (* ,scale ,name)))
			    
				    (font ;pax_font_sky
				      pax_font_saira_condensed
				      )
				    (text (text_.c_str))
				    (dims (pax_text_size font
							 font->default_size
							 text)))
			 
				(pax_draw_text buf
					       (hex #xffffffff) ; white
					       font
					       font->default_size
					       (/ (- buf->width
						     dims.x)
						  2.0)
					       (+ -10 (* .5 (+ ,graph-ymin
							       ,graph-ymax)))
					       text
					       )))
		    
			     (for-range (p fifoBME)
					(comments "draw measurements as points")
					(dotimes (i 3)
					  (dotimes (j 3)
					    (pax_set_pixel buf
							   (pax_col_hsv ,hue 180 200)
							   (+ i -1 (scaleTime p.x))
							   (+ j -1 (scaleHeight (dot p ,name)))))))
			     ))))))

	  ,(let* ((pitch-y (floor 240 4))
		  (graph-xmax 318s0)
		  (graph-ymin 1s0)
		  (graph-ymax (- pitch-y 1)))
	     
	     `(defun drawCO2 (buf )
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
				     (let ((res (* ,graph-xmax (/ (- x time_mi )
								  time_delta))))
				       (do0 (when (< res 1s0)
					      (setf res 1s0))
					    (when (< ,graph-xmax res)
					      (setf res ,graph-xmax)))
				       (return res))))
			(min_max_y (std--minmax_element (fifo.begin)
							(fifo.end)
							(lambda (p1 p2)
							  (declare (type "const Point2D&" p1 p2))
							  (return (< p1.y p2.y))
							  )))
			(min_y min_max_y.first->y)
			(max_y min_max_y.second->y)
			(scaleHeight (lambda
					 (v)
				       (declare (type float v)
						(capture "&")
						(values float))
				       (comments "v is in the range 400 .. 5000"
						 "map to 0 .. 239")
				       (let ((mi 400s0)
					     (ma (? (< max_y 1200s0) 1200s0 5000s0))
					     (res (+ ,graph-ymin
						     (* ,graph-ymax
							(- 1s0
							   (/ (- v mi)
							      (- ma mi)))))))
					 (when (< res ,graph-ymin)
					   (setf res ,graph-ymin))
					 (when (< ,graph-ymax res)
					   (setf res ,graph-ymax))
					 (return res))
				       ;; note: i think  to and bottom are inverted on screen 
				       )))
		    #+nil (dotimes (i (- (fifo.size) 1))
			    (let ((a (aref fifo i))
				  (b (aref fifo (+ i 1))))
			      (comments "draw measurements as line segments")
			      (pax_draw_line buf col
					     (scaleTime a.x)
					     (scaleHeight a.y)
					     (scaleTime b.x)
					     (scaleHeight b.y))))
		    (do0
		     (comments "write latest measurement")
		     (let (
			   (co2 (dot (aref fifo 0) y))
			   #+nil ,@(loop for e in `(temperature
						    humidity
						    pressure
						    gas_resistance)
					 collect
					 `(,e (dot (aref fifoBME 0) ,e)))
			     
			   (text_ (fmt--format
				   (string ;"co2={:4.0f} T={:2.1f} H={:2.1f}% p={:4.2f} R={:3.3f}"
				    "CO2={:4.0f}ppm"
				    )
				   co2
					;temperature
					; humidity
					; (/ pressure 1d3)
					;(/ gas_resistance 1d6)
				   ))
			   (font	;pax_font_sky
			     pax_font_saira_condensed
			     )
			   (text (text_.c_str))
			   (dims (pax_text_size font
						font->default_size
						text)))
			 
		       (pax_draw_text buf
				      (hex #xffffffff) ; white
				      font
				      font->default_size
				      (/ (- buf->width
					    dims.x)
					 2.0)
				      (+ -10 (* .5 (+ ,graph-ymin
						      ,graph-ymax)))
				      text
				      )))
		   
		    (for-range (p fifo)
			       (comments "draw measurements as points")
			       (dotimes (i 3)
				 (dotimes (j 3)
				   (pax_set_pixel buf
						  (pax_col_hsv 149 180 200)
						  (+ i -1 (scaleTime p.x))
						  (+ j -1 (scaleHeight p.y))))))

		    (progn
		      (let ((m 0d0)
			    (b 0d0)
			    (inliers ("std::vector<Point2D>"))
			    (hue 128)
			    (sat 255)
			    (bright 200)
			    (col (pax_col_hsv hue
					      sat bright))
			    )
			(ransac_line_fit fifo
					 m b inliers)
			(comments "draw the fit as line")
			(pax_draw_line buf col
				       (scaleTime time_mi)
				       (scaleHeight (+ b (* m time_mi)))
				       (scaleTime time_ma)
				       (scaleHeight (+ b (* m time_ma)))
				       )
			(comments "draw inliers as points")
			(for-range (p inliers)
				   (dotimes (i 3)
				     (dotimes (j 3)
				       (pax_set_pixel buf
						      (pax_col_hsv 0 255 255)
						      (+ i -1 (scaleTime p.x))
						      (+ j -1 (scaleHeight p.y))))))
			)

		     
		     

		      (do0
		       (comments "compute when a value of 1200ppm is reached")
		       (let ((x0 (/ (- 1200d0 b)
				    m))
			     (x0l (/ (- 500d0 b)
				     m)))

			 (progn
			   (let ((text_ (fmt--format (string "m={:3.4f} b={:4.2f} xmi={:4.2f} xma={:4.2f}")
						     m b time_mi time_ma ))
				 (text (text_.c_str))
				 (font pax_font_sky)
				 (dims (pax_text_size font
						      font->default_size
						      text)))
			     (pax_draw_text buf
					    (pax_col_hsv 160 128 128)
					;(hex #xffffffff) ; white
					    font
					    font->default_size
					    20
					    80
					    text)
			      
			     )
			   (progn
			     (let ((text_ (fmt--format (string "x0={:4.2f} x0l={:4.2f}")
						       x0 x0l))
				   (text (text_.c_str))
				   (font pax_font_sky)
				   (dims (pax_text_size font
							font->default_size
							text)))
			       (pax_draw_text buf
					      (pax_col_hsv 130 128 128)
					;(hex #xffffffff) ; white
					      font
					      font->default_size
					      20
					      60
					      text)
			      
			       ))
			   )
		   
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
						   (pax_col_hsv 30 128 128)
					; (hex #xffffffff) ; white
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
						   (pax_col_hsv 90 128 128)
					;(hex #xffffffff) ; white
						   font
						   font->default_size
						   20
						   140
						   text)
			      
				    )))))

		      )

		    ))
		))
	
	  (defun app_main ()
	    (ESP_LOGE TAG (string "welcome to the template app"))
	    #-nowifi
	    (do0
	     (let ((ret (nvs_flash_init)))
	       (when (logior (== ESP_ERR_NVS_NO_FREE_PAGES ret)
			     (== ESP_ERR_NVS_NEW_VERSION_FOUND ret))
		 (ESP_ERROR_CHECK (nvs_flash_erase))
		 (setf ret (nvs_flash_init)))
	       (ESP_ERROR_CHECK ret)
	       (ESP_LOGE TAG (string "esp wifi mode sta"))
	       (wifi_init_sta)
	       (connect_to_tcp_server) ))

	    (bsp_init) 
	  
	    (bsp_rp2040_init)
	    (setf buttonQueue (-> (get_rp2040)
				  queue))
	    (pax_buf_init &buf nullptr 320 240 PAX_BUF_16_565RGB)
					;(nvs_flash_init)
					;(wifi_init)
	  
	  
	    (uart_init)

	    (bsp_bme680_init)
	    (bme680_set_mode (get_bme680) BME680_MEAS_FORCED)
	    (bme680_set_oversampling (get_bme680)
				     BME680_OVERSAMPLING_X2
				     BME680_OVERSAMPLING_X2
				     BME680_OVERSAMPLING_X2
				     )
	  
	  
	    (while 1
		   (measureCO2)

		   (measureBME)
		 
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
						  (format nil "build ~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
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
		      
		     
		       ,@(loop for e in `(temperature
					  humidity
					  pressure
					;gas_resistance
					  )
			       collect
			       `(,(format nil "drawBME_~a" e)
				 &buf))
		     
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
			       (nowtext_ (fmt--format (string "now={:6.1f}")
						      now)
					;,(sprint :vars `(now))
					 ))
			   (pax_draw_text &buf
					  (hex #xffffffff) ; white
					  font
					  font->default_size
					  20
					  180
					  (nowtext_.c_str)))
			 #+nil (let (
				     (co2 (dot (aref fifo 0) y))
				     ,@(loop for e in `(temperature
							humidity
							pressure
							gas_resistance)
					     collect
					     `(,e (dot (aref fifoBME 0) ,e)))
			     
				     (text_ (fmt--format
					     (string ;"co2={:4.0f} T={:2.1f} H={:2.1f}% p={:4.2f} R={:3.3f}"
					      "co2={:4.0f} T={:2.1f} H={:2.1f}% p={:4.2f}"
					      )
					     co2
					     temperature
					     humidity
					     (/ pressure 1d3)
					;(/ gas_resistance 1d6)
					     ))
				     (font pax_font_sky
					;pax_font_saira_condensed
					   )
				     (text (text_.c_str))
				     (dims (pax_text_size font
							  font->default_size
							  text)))
			 
				 (pax_draw_text &buf
						(hex #xffffffff) ; white
						font
						font->default_size
						10
						#+nil (/ (- buf.width
							    dims.x)
							 2.0)
						200
						text
						)))

		       (drawCO2 &buf)
		     
		       (disp_flush)
		     
		       (let ((message (rp2040_input_message_t)))
			 (xQueueReceive buttonQueue
					&message
					2 ;10 ;portMAX_DELAY
					)

			 (when (logand (== RP2040_INPUT_BUTTON_HOME
					   message.input)
				       message.state)
			   (exit_to_launcher)))))))))))))



