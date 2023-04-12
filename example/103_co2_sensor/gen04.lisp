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
    (defparameter *source-dir* #P"example/103_co2_sensor/source04/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-py-generator
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (let ((n-fifo (floor 320 1))
	(l-data
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
    (let ((name `Wifi)
	  ;(members `((:name retry-attempts :type int :default 0)))
	  )
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-py-generator
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include freertos/FreeRTOS.h
				 freertos/task.h
				 freertos/event_groups.h
				 esp_wifi.h))
     :implementation-preamble
     `(do0
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
		
		(comments "event group should allow two different events"
			  "1) we are connected to access point with an ip"
			  "2) we failed to connect after a maximum amount of retries")
		))
       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h")))
     :code `(do0
	     (defclass ,name ()
	       "private:"
	       (space enum "{"
		(comma
		 (= WIFI_CONNECTED_BIT BIT0)
		 (= WIFI_FAIL_BIT BIT1)
		 (= EXAMPLE_ESP_MAXIMUM_RETRY 7))
		"}")
	       (space int s_retry_num)
	       (space EventGroupHandle_t s_wifi_event_group)
	       (defmethod event_handler (arg
					 event_base
					 event_id
					 event_data)
		 (declare (type void* arg event_data)
			  (type esp_event_base_t event_base)
			  (type int32_t event_id)
					;(static)
			  ;(values esp_event_handler_t)
			  )
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
	       "public:"
	       (defmethod ,name ()
		 (declare
		  (construct (s_retry_num 0)
			     )
		  (explicit)
		  ;(noexcept)
		  (values :constructor))
		 (do0
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
		 (reinterpret_cast<esp_event_handler_t> &Wifi--event_handler)
		 nullptr
		 &instance_any_id))
	       (ESP_ERROR_CHECK
		(esp_event_handler_instance_register
		 IP_EVENT
		 IP_EVENT_STA_GOT_IP
		  (reinterpret_cast<esp_event_handler_t> &Wifi--event_handler)
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
	       (vEventGroupDelete s_wifi_event_group)))))))))

    (let ((name `TcpConnection))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  )
       :implementation-preamble
       `(do0
	 (space "extern \"C\" "
		(progn
		  (include  "esp_wifi.h"
			    "esp_netif_types.h"
			    "nvs_flash.h"
			    "freertos/FreeRTOS.h"
			    "freertos/task.h"
			    "freertos/event_groups.h"
			    )
		  (include lwip/sockets.h)
		  (include<> arpa/inet.h)))
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h")))
       :code `(do0
	       (defclass ,name ()	 
		 "public:"
		 (defmethod ,name ()
		   (declare
		    (construct
		     )
		    (explicit)	    
		    (values :constructor))
		   (do0
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
			,(lprint :msg "failed to create socket"))
		      (when (!= 0 (connect sock
					   ("reinterpret_cast<const sockaddr*>" &addr)
					   (sizeof addr))) 
			,(lprint :msg "failed to connect to socket"))
		      ,(lprint :msg "connected to tcp server")
		      (let ((buffer_size 1024)
			    (read_buffer "std::array<char,buffer_size>{}"))
			(declare (type "constexpr auto" buffer_size))
			(let ((r (read sock
				       (read_buffer.data)
				       (- (read_buffer.size)
					  1))))
			  (when (< r 0)
			    ,(lprint :msg "failed to read data from socket"))
			  (setf (aref read_buffer r)
				(char "\\0"))
			  ,(lprint :msg "received data from server"
				   :vars `((read_buffer.data))))))))))))

    (let ((name `Ransac))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include DataTypes.h)
			  (include<> deque
		  random
		  vector
		  algorithm
		  cmath))
       :implementation-preamble
       `(do0
	 (include<> deque
		  random
		  vector
		  algorithm
		  cmath)
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h"))
	 ,@(loop for e in `(;(N_FIFO ,n-fifo)
			    (RANSAC_MAX_ITERATIONS ,(max n-fifo 12))
			    (RANSAC_INLIER_THRESHOLD 5.0 :type float)
			    (RANSAC_MIN_INLIERS 2))
	       collect
	       (destructuring-bind (name val &key (type 'int)) e
		 (format nil "const ~a ~a = ~a;" type name val))))
       :code `(do0
	       (defclass ,name ()
		 "private:"
		 (defmethod distance (p m b)
		   (declare (type Point2D p)
			    (type double m b)
			    (values double))
		   (return (/ (abs (- p.y
				      (+ (* m p.x)
					 b)))
			      (sqrt (+ 1 (* m m))))))
		 (defmethod ransac_line_fit (data m b inliers)
		   (declare (type "std::deque<Point2D>&" data)
			    (type "std::vector<Point2D>&" inliers)
			    (type double& m b))
		   
		   (when (< (data.size) 2)
		     (return))
		   "std::random_device rd;"
		   (comments "distrib0 must be one of the 5 most recent datapoints. i am not interested in fit's of the older data")
		   (let ((gen (std--mt19937 (rd)))
			 (distrib0 (std--uniform_int_distribution<> 0 5))
			 (distrib (std--uniform_int_distribution<> 0 (- (data.size)
									1)))
			 (best_inliers (std--vector<Point2D>))
			 (best_m 0d0)
			 (best_b 0d0))
		     (dotimes (i RANSAC_MAX_ITERATIONS)
		       (let ((idx1 (distrib gen))
			     (idx2 (distrib0 gen)))
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
			   inliers best_inliers)))
		 (space "std::vector<Point2D>" m_inliers)
		 (space "std::deque<Point2D>" m_data)
		 (space double m_m)
		 (space double m_b)
		 "public:"
		 (defmethod GetM ()
		   (declare 
			    (values double))
		   (return m_m))
		 (defmethod GetB ()
		   (declare (values double))
		   (return m_b))
		 (defmethod GetInliers ()
		   (declare (values "std::vector<Point2D>"))
		   (return m_inliers))
		 (defmethod ,name (data)
		   (declare
		    (type "std::deque<Point2D>" data)
		    (construct
		     (m_data data)
		     (m_inliers (std--vector<Point2D>)))
		    (explicit)
		    
		    (values :constructor))
		   (ransac_line_fit m_data m_m m_b m_inliers))))))


    (write-source
     (asdf:system-relative-pathname
      'cl-py-generator
      (merge-pathnames #P"DataTypes.h"
		       *source-dir*))
     `(do0
       "#pragma once"
      ; (space extern const char* TAG)
       ,@(loop for e in `((N_FIFO ,n-fifo)
					;(RANSAC_MAX_ITERATIONS ,(max n-fifo 12))
					;(RANSAC_INLIER_THRESHOLD 5.0 :type float)
					;(RANSAC_MIN_INLIERS 2)
			  )
	       collect
	       (destructuring-bind (name val &key (type 'int)) e
		 (format nil "const ~a ~a = ~a;" type name val)))
       (defstruct0 Point2D
	   (x double)
	 (y double))
	    (defstruct0 PointBME
		(x double)
	      (temperature double)
	      (humidity double)
	      (pressure double)
					;(gas_resistance double)
	      )))



    
    (let ((name `Display))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include<> string)
			  (space "extern \"C\" "
				 (progn
				   (do0
				    (include 
				     pax_gfx.h
				     pax_codecs.h
				     )
	   

				    )))		  
			  )
       :implementation-preamble
       `(do0
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h"))
	 (space "extern \"C\" "
	      (progn
		(do0
	   (include hardware.h
		    pax_gfx.h
		    pax_codecs.h
					;  ili9341.h
					; freertos/FreeRTOS.h
					; freertos/queue.h
		    esp_system.h
		    ;soc/rtc_cntl_reg.h
		   
		    
		    ;sys/time.h
		    )
	   

	   ))))
       :code `(do0
	       (defclass ,name ()	 
		 "private:"
		 (space pax_buf_t buf)
		 "public:"
		 (defmethod ,name ()
		   (declare
		    (construct
		     )
		    (explicit)	    
		    (values :constructor))
		   (pax_buf_init &buf nullptr 320 240 PAX_BUF_16_565RGB)
		   )
		 (defmethod background (hue sat bright)
		   (declare (type uint8_t hue sat bright))
		   (pax_background &buf (pax_col_hsv hue
						     sat bright)))
		 (defmethod set_pixel (x y &key (hue 128) (sat 0) (bright 0))
		   (declare (type uint8_t hue sat bright)
			    (type int x y))
		   (pax_set_pixel &buf
				  (pax_col_hsv hue sat bright)
				  x y)
		   )
		 (defmethod line (x0 y0 x1 y1 h s v)
			  (declare (type float x0 y0 x1 y1)
				   (type uint8_t h s v))
			  (pax_draw_line &buf (pax_col_hsv h s v)
				       x0 y0 x1 y1))
		 (defmethod small_text (str &key (x "-1.0f") (y "-1.0f") (h 128) (s 255) (v 255))
		   (declare (type std--string str)
			    (type uint8_t h s v)
			    (type float x y))
		   (text str pax_font_sky x y h s v))
		 (defmethod large_text (str &key (x "-1.0f") (y "-1.0f") (h 128) (s 255) (v 255))
		   (declare (type std--string str)
			    (type uint8_t h s v)
			    (type float x y))
		   (text str pax_font_saira_condensed x y h s v))
		 (defmethod text (str font x y h s v)
		   (declare (type std--string str)
			    (type uint8_t h s v)
			    (type float x y)
			    (type "const pax_font_t*" font))
		   (let ((text_ (str.c_str))
			 (dims (pax_text_size font
					      font->default_size
					      text_))
			 (x_ (/ (- buf.width
				   dims.x)
				2.0))
			 (y_ (/ (- buf.height
				   dims.y)
				2.0)))
		     (comments "center coordinate if x or y < 0")
		     (when (< 0 x)
		       (setf x_ x))
		     (when (< 0 y)
		       (setf y_ y))
		     (pax_draw_text &buf
				    (pax_col_hsv h s v)
				    font
				    font->default_size
				    x_ 
				    y_
				    text_)))
		 (defmethod flush ()
		   (ili9341_write (get_ili9341)
					; buf.buf
				  #-nil
				  ("static_cast<const uint8_t*>" buf.buf))))
	       )))


    (let ((name `Uart))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include DataTypes.h)
			  (include<> deque)
			   (space "extern \"C\" "
	      (progn
		(do0
		 (include driver/uart.h))))
			  )
       :implementation-preamble
       `(do0
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h")
	  (space "extern \"C\" "
	      (progn
		(do0
		 (include<> esp_log.h)
		 (include driver/uart.h))))
	  
	  ))
       :code `(do0
	       (defclass ,name ()	 
		 "private:"
		 "static constexpr char*TAG = \"mch2022-co2-uart\";"
		 (space static constexpr uart_port_t (= CO2_UART UART_NUM_1))
		 (space static constexpr size_t (=
						 BUF_SIZE UART_FIFO_LEN))
		 "public:"
		 (defmethod measureCO2 (fifo)
		   (declare (type "std::deque<Point2D>&" fifo))
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
		 (defmethod ,name ()
		   (declare
		    (construct
		     )
		    (explicit)	    
		    (values :constructor))
		   (do0
		    (ESP_LOGE TAG (string "initialize uart"))
		    (when (uart_is_driver_installed CO2_UART) 
		      (return))

		    (unless (== ESP_OK (uart_set_pin CO2_UART
						     27 ;; tx
						     39 ;; rx
						     UART_PIN_NO_CHANGE
						     UART_PIN_NO_CHANGE))

		      (ESP_LOGE TAG (string "error: uart_set_pin 27 39")))
		    (unless (== ESP_OK
				(uart_driver_install CO2_UART
						     200 ;BUF_SIZE ;; rx
						     0 ;BUF_SIZE ;; tx
						     0 ;; queue length
 						     nullptr ;; queue out
						     0 ;; interrupt
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
			(ESP_LOGE TAG (string "error: uart_param_config"))))))))))

    (let ((name `BmeSensor))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include DataTypes.h)
			  (include<> deque)
			   (space "extern \"C\" "
				  (progn
				    (do0
				    
				     (include bme680.h)
				     ))))
       :implementation-preamble
       `(do0
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h")
	  (space "extern \"C\" "
	      (progn
		(do0
		 (include<> esp_log.h
			    bme680.h
			    hardware.h)
		 )))))
       :code `(do0
	       (defclass ,name ()	 
		 "private:"
		 "static constexpr char*TAG = \"mch2022-co2-bme\";"
		 
		 "public:"
		 (defmethod measureBME (fifoBME fifo)
		   (declare (type "std::deque<PointBME>&" fifoBME)
			    (type "std::deque<Point2D>&" fifo))
		   (do0
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

		      (ESP_LOGE TAG (string "%s")
				(dot ,(sprint  :vars `((== bme nullptr)
						       temperature humidity pressure ;gas_resistance
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
			  (fifoBME.push_front p))))))
		 (defmethod ,name ()
		   (declare
		    (construct)
		    (explicit)	    
		    (values :constructor))
		   (do0 (bsp_bme680_init)
		 (bme680_set_mode (get_bme680) BME680_MEAS_FORCED)
		 (bme680_set_oversampling (get_bme680)
					  BME680_OVERSAMPLING_X2
					  BME680_OVERSAMPLING_X2
					  BME680_OVERSAMPLING_X2
					  ))
		   ))
	       )))

    (let ((name `Graph))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include DataTypes.h
				   Display.h)
			  (include<> deque)
			  #+nil (space "extern \"C\" "
				  (progn
				    (do0
				    
				     (include bme680.h)
				     ))))
       :implementation-preamble
       `(do0
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include core.h
		   Ransac.h)
	  (include<> deque)
	  #+nil
	  (space "extern \"C\" "
		 (progn
		   (do0
		    (include<> esp_log.h
			       bme680.h
			       hardware.h)
		    )))))
       :code `(do0
	       (defclass ,name ()	 
		 "private:"
		 "Display& m_display;"
		  "std::deque<Point2D>& m_fifo; "
		  "std::deque<PointBME>& m_fifoBME; "

		 
		 "public:"

		 ,(let* ((pitch-y (floor 240 4))
		  (graph-xmax 318s0)
		  (graph-ymin 1s0)
		  (graph-ymax (- pitch-y 1)))
	     
	     `(defmethod carbon_dioxide ()
		
		(when (< (m_fifo.size) 2)
		  (return))
		(let ((hue 12)
		      (sat 255)
		      (bright 255)
		      (col (pax_col_hsv hue
					sat bright))
		      )
		  (let ((time_ma (dot (aref m_fifo 0) x))
			(time_mi (dot (aref m_fifo (- (dot m_fifo (size)) 1))
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
			(min_max_y (std--minmax_element (m_fifo.begin)
							(m_fifo.end)
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
		    (do0
		     (comments "write latest measurement")
		     (let (
			   (co2 (dot (aref m_fifo 0) y))
			     )

		       (m_display.large_text (fmt--format
				   (string 
				    "CO2={:4.0f}ppm"
				    )
				   co2)
					   -1 (+ -10 (* .5 (+ ,graph-ymin
							      ,graph-ymax))))))
		   
		    (for-range (p m_fifo)
			       (comments "draw measurements as points")
			       (dotimes (i 3)
				 (dotimes (j 3)
				   (m_display.set_pixel
				    (+ i -1 (scaleTime p.x))
				    (+ j -1 (scaleHeight p.y))
				    149 180 200
				    ))))

		    (progn
		      (let ((ransac (Ransac m_fifo))
			    (m (ransac.GetM))
			    (b (ransac.GetB))
			    (inliers (ransac.GetInliers))
			    (hue 128)
			    (sat 255)
			    (bright 200)
			    (col (pax_col_hsv hue
					      sat bright)))
			
			(comments "draw the fit as line")
			
			(m_display.line (scaleTime time_mi)
				      (scaleHeight (+ b (* m time_mi)))
				      (scaleTime time_ma)
				      (scaleHeight (+ b (* m time_ma)))
				      188 255 200)
			(comments "draw inliers as points")
			(for-range (p inliers)
				   (dotimes (i 3)
				     (dotimes (j 3)
				       (m_display.set_pixel
					(+ i -1 (scaleTime p.x))
					(+ j -1 (scaleHeight p.y))
					0 255 255)))))

		     
		     

		      (do0
		       (comments "compute when a value of 1200ppm is reached")
		       (let ((x0 (/ (- 1200d0 b)
				    m))
			     (x0l (/ (- 500d0 b)
				     m)))

			 
			 (m_display.small_text
			  (fmt--format (string "m={:3.4f} b={:4.2f} xmi={:4.2f} xma={:4.2f}")
				       m b time_mi time_ma )
			  20 80
			  160 128 128
			  )
			 (m_display.small_text
			  (fmt--format (string "x0={:4.2f} x0l={:4.2f}")
				       x0 x0l)
			  20 60 130 128 128)
		   
			 (if (< time_ma x0)
			     (do0 (comments "if predicted intersection time is in the future, print it")
				  (let ((time_value (static_cast<int> (- x0 time_ma)))
					(hours (int (/ time_value 3600)))
					(minutes (int (/ (% time_value 3600) 60)))
					(seconds (% time_value 60))
					(text_ (fmt--format (string "air room in (h:m:s) {:02d}:{:02d}:{:02d}")
							    hours minutes seconds))
					)
				    (m_display.small_text
				     text_
				     20 140
				     30 128 128)))
			     (do0 (comments "if predicted intersection time is in the past, then predict when airing should stop")
				  (let ((x0 (/ (- 500d0 b)
					       m))
					(time_value (static_cast<int> (- x0 time_ma)))
					(hours (int (/ time_value 3600)))
					(minutes (int (/ (% time_value 3600) 60)))
					(seconds (% time_value 60))
					(text_ (fmt--format (string "air of room should stop in (h:m:s) {:02d}:{:02d}:{:02d}")
							    hours minutes seconds))
					)

				    (m_display.small_text
				     text_
				     20 140
				     90 128 128)))))))))))
		 
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
			       `(defmethod ,(format nil "~a" name)
				    (
				     )
				  (let ((time_ma (dot (aref m_fifoBME 0) x))
					(time_mi (dot (aref m_fifoBME (- (dot m_fifoBME (size)) 1))
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
					(min_max_y (std--minmax_element (m_fifoBME.begin)
									(m_fifoBME.end)
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
				     (let ((,name (dot (aref m_fifoBME 0) ,name)))
				       (m_display.large_text
					(fmt--format
					 (string ;"co2={:4.0f} T={:2.1f} H={:2.1f}% p={:4.2f} R={:3.3f}"
					  ,(format nil "~a=~a~a" short-name fmt unit))
					 (* ,scale ,name))
					-1
					(+ -10 (* .5 (+ ,graph-ymin
							,graph-ymax))))))
		    		    (for-range (p m_fifoBME)
					       (comments "draw measurements as points")
					       (dotimes (i 3)
						 (dotimes (j 3)
						   (m_display.set_pixel (+ i -1 (scaleTime p.x))
									(+ j -1 (scaleHeight (dot p ,name)))

									,hue 180 200))))
				    ))))))
		 (defmethod ,name (display fifo fifoBME)
		   (declare
		    (type Display& display)
		    (type "std::deque<Point2D>&" fifo)
		    (type "std::deque<PointBME>&" fifoBME)
		    (construct (m_display display)
			       (m_fifo fifo)
			       (m_fifoBME fifoBME)
			       )
		    (explicit)	    
		    (values :constructor))
		   
		   ))
	       )))
    
    
    (write-source
     (asdf:system-relative-pathname
      'cl-py-generator
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0
       
       (include<> deque
		;  random
		 ; vector
		 ; algorithm
		  ;cmath
		  )
       (include   Wifi.h
		  Graph.h
		  BmeSensor.h
		  Uart.h
		  Display.h
		  TcpConnection.h
		  DataTypes.h
		  ;Ransac.h
		  )
       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h"))


       
       
       "std::deque<Point2D> fifo; "	;(N_FIFO,{0.0,0.0});
       "std::deque<PointBME> fifoBME; " ;(N_FIFO,{0.0,0.0,0.0,0.0});


       
       (space
	"extern \"C\" "
	(progn

	  (do0
	   (include hardware.h
		    ;pax_gfx.h
		    ;pax_codecs.h
					;  ili9341.h
					; freertos/FreeRTOS.h
					; freertos/queue.h
		   ; esp_system.h
					;nvs.h
					;nvs_flash.h
					;wifi_connect.h
					;wifi_connection.h
					; soc/rtc.h
		    soc/rtc_cntl_reg.h
		    
					;gpio_types.h
		    ;driver/uart.h
		  ;  sys/time.h
		    )
	   (include nvs_flash.h)

	   (include<> esp_log.h)

	   )
	  ;; here they define another uart, uart0:
	  ;; ../components/mch2022-rp2040/rp2040bl.c


	  (do0 "static const char*TAG = \"mch2022-co2-app\";"
	       
	       "xQueueHandle buttonQueue;")

	  

	  (defun exit_to_launcher ()
	    (REG_WRITE RTC_CNTL_STORE0_REG 0)
	    (esp_restart))

					;"#define CO2_UART UART_NUM_1"
					;"#define BUF_SIZE UART_FIFO_LEN" ;; 128

	  ;;../esp-idf/docs/en/api-reference/peripherals/uart.rst
	
	  

	  
	
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
	       "Wifi wifi;"		;(wifi_init_sta)
	       "TcpConnection tcp;"	; (connect_to_tcp_server)
	       ))

	    

	    (bsp_init) 
	  
	    (bsp_rp2040_init)
	    (setf buttonQueue (-> (get_rp2040)
				  queue))

	    (space Display display)
	    (space Graph (graph display fifo fifoBME))
	    	  
	    (space Uart uart)
	    (space BmeSensor bme)
	    
	  
	  
	    (while 1
		   (uart.measureCO2 fifo)

		   (bme.measureBME fifoBME fifo)
		 
		   (display.background 129 0 0)

		   
		   ,@(loop for e in `(temperature
				      humidity
				      pressure
					;gas_resistance
				      )
			   collect
			   `(dot graph (,(format nil "~a" e))))
		   

		   (display.small_text ,(sprint :msg (multiple-value-bind
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
		   
		   (progn
		     (let ((now (dot (aref fifo 0) x)))
		       (display.small_text (fmt--format (string "now={:6.1f}")
							now)
					   20 180)))

		   (graph.carbon_dioxide)
		   
		   (display.flush)
		   
		   (let ((message (rp2040_input_message_t)))
		     (xQueueReceive buttonQueue
				    &message
				    2	;10 ;portMAX_DELAY
				    )

		     (when (logand (== RP2040_INPUT_BUTTON_HOME
				       message.input)
				   message.state)
		       (exit_to_launcher)))))))))))



