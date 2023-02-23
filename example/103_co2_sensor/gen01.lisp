(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/103_co2_sensor/source01/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-py-generator
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)


  (write-source
   (asdf:system-relative-pathname
    'cl-py-generator
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0




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
		  nvs.h
		  nvs_flash.h
		  wifi_connect.h
		  wifi_connection.h
		  soc/rtc.h
		  soc/rtc_cntl_reg.h)

	 (include<> esp_log.h))

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

	(defun uart_init ()
	  (let ((BUF_SIZE 1024)
		(config (uart_config_t
			 (designated-initializer
			  :baud_rate 9600
			  :data_bits UART_DATA_8_BITS
			  :parity UART_PARITY_DISABLE
			  :stop_bits UART_STOP_BITS_1
			  :flow_ctrl UART_HW_FLOWCTRL_DISABLE
			  :source_clk UART_SCLK_APB))))
	    (uart_driver_install UART_NUM_1 (* BUF_SIZE 2)
				 0 0 nullptr 0)
	    (uart_param_config UART_NUM_1 &config)
	    (uart_set_pin UART_NUM_1
			  GPIO_NUM_4
			  GPIO_NUM_0
			  UART_PIN_NO_CHANGE
			  UART_PIN_NO_CHANGE)))
	
	(defun app_main ()
	  (ESP_LOGI TAG (string "welcome to the template app"))
	  (bsp_init)
	  
	  (bsp_rp2040_init)
	  (setf buttonQueue (-> (get_rp2040)
				queue))
	  (pax_buf_init &buf nullptr 320 240 PAX_BUF_16_565RGB)
	  (nvs_flash_init)
	  (wifi_init)
	  (uart_init)
	  (while 1
		 (let ((hue (and (esp_random)
				 255
				 ))
		       (sat 255)
		       (bright 255)
		       (col (pax_col_hsv hue
					 sat bright))
		       )
		   (pax_background &buf col)
		   (let ((text (string "hello martin"))
			 (font pax_font_saira_condensed)
			 (dims (pax_text_size font
					      font->default_size
					      text)))
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
				      portMAX_DELAY)
		       (when (logand (== RP2040_INPUT_BUTTON_HOME
					 message.input)
				     message.state)
			 (exit_to_launcher))))))))))))



