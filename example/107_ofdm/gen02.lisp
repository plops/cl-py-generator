(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  #+nil
  (progn
    (defparameter *source-dir*       "/home/martin/src/my_fancy_app_name/main/")
    (defparameter *full-source-dir*  "/home/martin/src/my_fancy_app_name/main/"))
  #-nil
  (progn
    (defparameter *source-dir* #P"example/107_ofdm/source02/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-py-generator
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (let ()

    (write-source
     (asdf:system-relative-pathname
      'cl-py-generator
      (merge-pathnames #P"OfdmConstants.h"
		       *source-dir*))
     `(do0
       "#pragma once"
       (include<> cstddef
		  complex)
       ,@(loop for e in `((:name PI :value ;3.14159265358979323846
				 ,(format nil "~3,12f" pi)
				 :type double
				 )
			  (:name FFT_SIZE :value 64 :type size_t)
			  (:name CP_SIZE :value 16 :type size_t)
			  (:name SYMBOLS :value 10 :type size_t))
	       collect
	       (destructuring-bind (&key name value type) e
		 (format nil "constexpr ~a ~a = ~a;" type name value))
	       )
       (space using (= Cplx std--complex<double>))
       )) 
    (let ((name `OfdmTransmitter)
					
	  )
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include OfdmConstants.h)
			  (include<> vector))
       :implementation-preamble
       `(do0
	 

	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h"))
	 (include<>
	  iostream
	  vector
	  complex
	  cmath
	  algorithm
	  random
	  chrono
	  cassert
	  array
	  fftw3.h)
	 
	 )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 (defmethod ,name ()
		   (declare
		    (construct		; (s_retry_num 0)
		     )
		    (explicit)
					;(noexcept)
		    (values :constructor))
		 )
		 (defmethod transmit (data)
		   (declare (type "const std::vector<Cplx>&" data)
			    (values "std::vector<Cplx>"))
		   
		   (let ((preamble (generatePreamble))
			 ;; fixme: do i need to allocate this with a fftw3 function?
			 (in ("std::array<Cplx,FFT_SIZE>"))
			 (out ("std::array<Cplx,FFT_SIZE>"))
			 (fftPlan (fftw_plan_dft_1d FFT_SIZE
						    (reinterpret_cast<fftw_complex*>
						     (in.data))
						    (reinterpret_cast<fftw_complex*>
						     (out.data))
						    FFTW_FORWARD
						    FFTW_ESTIMATE))
			 (ifftData (std--vector<Cplx> (* FFT_SIZE SYMBOLS))))
		     (dotimes (symbol SYMBOLS)
		       (dotimes (i FFT_SIZE)
			 (setf (aref in i)
			       (aref data (+ i (* symbol FFT_SIZE)))))
		       (do0 (fftw_execute fftPlan)
			    (std--copy (out.data) (+ (out.data) FFT_SIZE)
				       (+ (ifftData.begin)
					  (* symbol FFT_SIZE)))))
		     (comments "insert cyclic prefix and preamble"
			       )
		     (let ((transmittedData (std--vector<Cplx> (+ (* SYMBOLS (+ FFT_SIZE
									      CP_SIZE))
								  (* 2 (+ FFT_SIZE
									  CP_SIZE))))))
		       (std--copy (preamble.begin)
				  (preamble.end)
				  (transmittedData.begin))
		       (dotimes (symbol SYMBOLS)
			 (std--copy (+ (ifftData.begin)
				       (* symbol FFT_SIZE))
				    (+ (ifftData.begin)
				       (* symbol FFT_SIZE)
				       CP_SIZE)
				    (+ (transmittedData.begin)
				       (* 2 (+ FFT_SIZE
					       CP_SIZE))
				       (* symbol (+ FFT_SIZE
						    CP_SIZE))))
			 (std--copy (+ (ifftData.begin)
				       (* symbol FFT_SIZE)
				       CP_SIZE)
				    (+ (ifftData.begin)
				       (* (+ 1 symbol) FFT_SIZE)
				       )
				    (+ (transmittedData.begin)
				       (* 2 (+ FFT_SIZE
					       CP_SIZE))
				       (* symbol (+ FFT_SIZE
						    CP_SIZE))
				       CP_SIZE)))
		       (fftw_destroy_plan fftPlan)
		       (return transmittedData))
		     ))
		 "private:"
		 (defmethod generatePreamble ()
		     (declare 
			      (values "std::vector<Cplx>"))
		   (let ((preamble (std--vector<Cplx> (* 2 (+ FFT_SIZE CP_SIZE))))
			 (random_symbols (std--vector<Cplx> (/ FFT_SIZE 2)))
			 (generator (std--default_random_engine))
			 (distribution (std--uniform_real_distribution<double> -1 1))
			 )
		     (dotimes (i (/ FFT_SIZE 2))
		       (setf (aref random_symbols i)
			     (Cplx (distribution generator)
				   (distribution generator))))

		     (let (
			 ;; fixme: do i need to allocate this with a fftw3 function?
			 (in ("std::array<Cplx,FFT_SIZE>"))
			 (out ("std::array<Cplx,FFT_SIZE>"))
			 (ifftPlan (fftw_plan_dft_1d FFT_SIZE
						    (reinterpret_cast<fftw_complex*>
						     (in.data))
						    (reinterpret_cast<fftw_complex*>
						     (out.data))
						    FFTW_BACKWARD
						    FFTW_ESTIMATE)))
		       (dotimes (i (/ FFT_SIZE 2))
			 (setf (aref in i) (aref random_symbols i)
			       (aref in (+ i (/ FFT_SIZE
						2)))
			       (std--conj (aref random_symbols i))))
		       (fftw_execute ifftPlan)
		       (comments "Add cyclic prefix and copy repeated preambels")
		       (std--copy (+ (out.data) FFT_SIZE -CP_SIZE)
				  (+ (out.data) FFT_SIZE)
				  (preamble.begin))
		       (std--copy (out.data)
				  (+ (out.data) FFT_SIZE)
				  (+ (preamble.begin)
				     CP_SIZE))
		       (std--copy (preamble.begin)
				  (+ (preamble.begin)
				     FFT_SIZE
				     CP_SIZE)
				  (+ (preamble.begin)
				     FFT_SIZE
				     CP_SIZE))
		       (fftw_destroy_plan ifftPlan)
		       (return preamble)
		     ))))
	       )))

    (let ((name `OfdmReceiver)
					
	  )
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include OfdmConstants.h)
			  (include<> vector))
       :implementation-preamble
       `(do0
	 

	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h"))
	 (include<>
	  iostream
	  vector
	  complex
	  cmath
	  algorithm
	  random
	  chrono
	  cassert
	  array
	  fftw3.h)
	 
	 )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 (defmethod ,name ()
		   (declare
		    (construct		; (s_retry_num 0)
		     )
		    (explicit)
					;(noexcept)
		    (values :constructor))
		 )
		 (defmethod receive (receivedData)
		   (declare (type "const std::vector<Cplx>&" receivedData)
			    (values "std::vector<Cplx>"))
		   
		   (let ((start_index (schmidlCoxSynchronization receivedData))
			 ;; fixme: do i need to allocate this with a fftw3 function?
			 (in ("std::array<Cplx,FFT_SIZE>"))
			 (out ("std::array<Cplx,FFT_SIZE>"))
			 (fftPlan (fftw_plan_dft_1d FFT_SIZE
						    (reinterpret_cast<fftw_complex*>
						     (in.data))
						    (reinterpret_cast<fftw_complex*>
						     (out.data))
						    FFTW_FORWARD
						    FFTW_ESTIMATE))
			 (fftData (std--vector<Cplx> (* FFT_SIZE SYMBOLS))))
		     
		     (dotimes (symbol SYMBOLS)
		       (std--copy (+ (receivedData.begin)
				     start_index
				     (* symbol (+ FFT_SIZE
						  CP_SIZE))
				     CP_SIZE)
				  (+ (receivedData.begin)
				     start_index
				     (* (+ 1 symbol) (+ FFT_SIZE
						    CP_SIZE)))
				  (in.data))
		       (do0 (fftw_execute fftPlan)
			    (std--copy (out.data) (+ (out.data) FFT_SIZE)
				       (+ (fftData.begin)
					  (* symbol FFT_SIZE)))))
		     (fftw_destroy_plan fftPlan)
		     (return fftData)
		     ))
		 "private:"
		 (defmethod schmidlCoxSynchronization (receivedData)
		   (declare (type "const std::vector<Cplx>&" receivedData)
			    (values "size_t"))
		   
		   (let ((R (std--vector<double> (+ FFT_SIZE CP_SIZE)
						 "0.0"))
			 (M (std--vector<double> (+ FFT_SIZE CP_SIZE)
						 "0.0"))
			 (P (double "0.0"))
			 
			 )
		     (dotimes (i (+ FFT_SIZE CP_SIZE))
		       (setf (aref R i)
			     (std--abs (* (aref receivedData (+ i FFT_SIZE CP_SIZE))
					  (std--conj (aref receivedData i)))))
		       (setf (aref M i)
			     (+
			      (std--norm (aref receivedData (+ i FFT_SIZE CP_SIZE))
					 )
			      (std--norm (aref receivedData i))))
		       (incf P (aref M i))
		       )
		     (let ((max_metric (double "-1.0"))
			     (start_index (size_t 0)))
		       (dotimes (i (+ FFT_SIZE
				      CP_SIZE))
			 (let ((metric (/ (aref R i)
					  (/ (aref M i)
					     P))))
			   (when (< max_metric
				    metric)
			     (setf max_metric metric
				   start_index i))))
		       (return start_index)))))
	       )))
    
    (write-source
     (asdf:system-relative-pathname
      'cl-py-generator
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0
       
       (include<> random
		  vector
					; algorithm
					;cmath
		  )
       (include   OfdmTransmitter.h
		  OfdmReceiver.h)
       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h"))


       
       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int))
	 (let ((data (std--vector<Cplx> (* FFT_SIZE SYMBOLS)))
	       (generator (std--default_random_engine))
	       (distribution (std--uniform_real_distribution<double> "-1.0"
								     "1.0")))
	   (dotimes (i (* FFT_SIZE SYMBOLS))
	     (setf (aref data i)
		   (Cplx (distribution generator)
			 (distribution generator))))
	   (let ((transmitter (OfdmTransmitter))
		 (transmittedData (transmitter.transmit data))
		 (receiver (OfdmReceiver))
		 (receivedData (receiver.receive transmittedData)))
	     (let ((err (double "0.0"))
		   (avgPower (double "0.0")))
	       (dotimes (i (* SYMBOLS FFT_SIZE))
		 (incf err (std--norm (- (aref data i)
					 (aref receivedData i))))
		 (incf avgPower (std--norm (aref data i))))
	       (setf err (/ err
			    (* FFT_SIZE
			       SYMBOLS)))
	       (setf avgPower (/ avgPower
				 (* FFT_SIZE
				    SYMBOLS)))
	       ;; normalized mean squared error
	       (let ((mse (std--sqrt err))
		     (nmse (/ err avgPower)))
		 ,(lprint :vars `(mse nmse)))
	       (return 0)))))
       
       ))))



