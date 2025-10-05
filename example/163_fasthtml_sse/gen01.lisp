(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:my-py-project
  (:use #:cl #:cl-py-generator)) 

(in-package #:my-py-project)

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p01_top"
						 "example/163_fasthtml_sse/"))
 `(do0
   (imports (random time asyncio))
   (imports-from (loguru logger)
		 (fasthtml.common *))
          
   (do0
    (logger.remove)
    (logger.add
     sys.stdout
     :format (string "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>")
     :colorize True
     :level (string "DEBUG")
					;:utc True
     ))

    (logger.info (string "Logger configured"))


   (comments "https://www.npmjs.com/package/htmx-ext-sse https://x.com/jeremyphoward/status/1829897065440428144/photo/1")

   (setf hdrs (tuple (Script :src (string "https://unpkg.com/htmx-ext-sse@2.2.3/sse.js")))
	 (ntuple app rt) (fast_app :hdrs hdrs))

   (do0
    @rt
    (def index ()
      (return (ntuple
	       (Titled (string "SSE Random Number Generator")
		       (P (string "Generate pairs of random numbers, as the list grows scroll downwards."))
		       (Div :hx_ext (string "sse")
			    :sse_connect (string "/number-stream")
			    :hx_swap (string "beforeend show:bottom")
			    :sse_swap (string "message")))))))

   (setf shutdown_event (signal_shutdown))
   (space async (def number_generator ()
		  (while (not (shutdown_event.is_set))
			 (setf data (Div (Article (random.randint 1 100))
					 (Article (random.randint 1 100))))
			 (yield (sse_message data))
			 (await (asyncio.sleep 1)))))

   (@rt (string "/number-stream"))
   (space async (def get ()
		  (return (EventStream (number_generator)))))
   (serve)
))


