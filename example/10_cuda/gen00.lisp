(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)
(defparameter *cuda-code*
  (cl-cpp-generator2::emit-c
   :code
   `(do
     (defun doublify (a)
       (declare (values "__global__ void")
		(type "float*" a))
       (let ((idx (+ threadIdx.x
		     (* 4 threadIdx.y))))
	 (declare (type int idx))
	 (*= (aref a idx) 2))))))

(in-package :cl-py-generator)

(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/10_cuda")
  (defparameter *code-file* "run_00_double")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# https://documen.tician.de/pycuda/tutorial.html"
	    (do0
	     #+nil (imports (matplotlib))
			      ;(matplotlib.use (string "Agg"))
			      (imports ((plt matplotlib.pyplot)))
			 (plt.ion))
	    
	    (imports (			;os
					;sys
					;time
					;docopt
					;pathlib
			    (np numpy)
					;serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
			    (cuda pycuda.driver)
			    pycuda.autoinit
			    ))
	    "from pycuda.compiler import SourceModule"
	    (do0
	     (setf a (dot (np.random.randn 4 4)
			  (astype np.float32))
		   a_gpu (cuda.mem_alloc a.nbytes)
		   )
	     (cuda.memcpy_htod a_gpu a)
	     (setf mod (SourceModule
			(string3 ,cl-cpp-generator2::*cuda-code*))
		   
		   func (mod.get_function (string "doublify"))
		   
		   )
	     (func a_gpu :block (tuple 4 4 1))
	     (setf a_doubled (np.empty_like a))
	     (cuda.memcpy_dtoh a_doubled a_gpu)
	     (print a_doubled)
	     (print a))
	    
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
