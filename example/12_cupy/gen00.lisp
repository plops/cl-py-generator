(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *c-code* (cl-cpp-generator2::emit-c
			       :code
			       `(do)))
 )



(in-package :cl-py-generator)

(progn
  
  
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/12_cupy")
  (defparameter *code-file* "run_00_cupy")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (let* ((code
	  `(do0
	    "# https://youtu.be/CQDsT81GyS8?t=3238 Valentin Haenel: Create CUDA kernels from Python using Numba and CuPy | PyData Amsterdam 2019"
	    "# pip3 install --user cupy-cuda101"
	    "# pip3 install --user numba"
	    "#  sudo ln -s /opt/cuda/nvvm/lib64/libnvvm.so* /usr/lib"
	    #+nil (do0
	     (imports (matplotlib))
					;(matplotlib.use (string "Agg"))
	     (imports ((plt matplotlib.pyplot)))
	     (plt.ion))
	    
	     (imports (			;os
					;sys
					;time
					;docopt
					;pathlib
		       (np numpy)
		       (cp cupy)
					;serial
					;(pd pandas)
					;(xr xarray)c
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      math
		       ))
	     "from numba import vectorize"
	     (do0
	      (setf ary (dot (cp.arange 10)
			     (reshape (tuple 2 5))))
	      (print (repr ary))
	      ,@(loop for e in `(dtype shape strides device) collect
		     `(print (dot (string ,(format nil "~a = {}" e))
				  (format (dot ary ,e)))))


	      
	      ;; asarray  .. move to gpu
	      ;; asnumpy  .. move to cpu

	      (do0
	       "# numba on gpu"
	       (@vectorize (list (string "int64(int64,int64)")) 
			   :target (string "cuda"))
	       (def add_ufunc (x y)
		 (return (+ x y)))
	       (setf a (np.array (list 1 2 3 4))
		     b (np.array (list 10 20 30 40))
		     b_col (aref b ":" np.newaxis)
		     c (dot (np.arange (* 4 4))
			    (reshape (tuple 4 4))))
	       (print (add_ufunc a b))
	       (print (add_ufunc b_col c))
	      )
	    ))))
    (cl-py-generator::write-source (format nil "~a/source/~a" *path* *code-file*) code)))


 
