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
       ,@(loop for e in `((:name PI :value #.pi :type double)
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
					;(members `((:name retry-attempts :type int :default 0)))
	  )
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-py-generator
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include OfdmConstants.h))
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
			    (std--copy out (+ out FFT_SIZE)
				       (+ (ifftData.begin)
					  (* symbol FFT_SIZE)))))
		     ))
		 "private:"
		 (defmethod generatePreamble ()
		     (declare 
			      (values "std::vector<Cplx>"))))
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
       (include   OfdmTransmitter.h)
       (do0
	"#define FMT_HEADER_ONLY"
	(include "core.h"))


       
       (defun main (argc argv)
	 (declare (type int argc)
		  (type char** argv)
		  (values int)))
       
       ))))



