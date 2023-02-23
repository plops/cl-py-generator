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
		  soc/rtc_cntl_reg.h

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

	"#define CO2_UART UART_NUM_2"
	"#define BUF_SIZE 100"

	;;../esp-idf/docs/en/api-reference/peripherals/uart.rst
	
	(defun uart_init ()

	  (when (uart_is_driver_installed CO2_UART)
	    (return))
	  (uart_set_pin CO2_UART
			  4 5 
			  UART_PIN_NO_CHANGE
			  UART_PIN_NO_CHANGE)
	  (uart_driver_install CO2_UART (* BUF_SIZE 2)
			       0 0 nullptr 0)
	  (let (
		(config (uart_config_t
			 (designated-initializer
			  :baud_rate 9600
			  :data_bits UART_DATA_8_BITS
			  :parity UART_PARITY_DISABLE
			  :stop_bits UART_STOP_BITS_1
			  :flow_ctrl UART_HW_FLOWCTRL_DISABLE
			  :source_clk UART_SCLK_APB))))
	    
	    (ESP_ERROR_CHECK (uart_param_config CO2_UART &config))
	    ))
	
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

	  (progn
	    ,(let ((l `(#xff #x01 #x86 0 0 0 0 0 #x79)))
	     `(let ((command (curly ,@(loop for e in l
					     collect
					     `(hex ,e))))
		    (response)
		    )
		(declare (type (array "unsigned char" ,(length l)) command response))
		(uart_write_bytes CO2_UART command ,(length l))
		(let ((l (uart_read_bytes CO2_UART response ,(length l)
					  100)))
		  (when (== 9 l)
		    (when (logand (== #xff (aref response 0))
				  (== #x86 (aref response 1)))
		      (let ((co2 (+ (* 256 (aref data 2))
				    (aref data 3)))))))))))
	  
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



