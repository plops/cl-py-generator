(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/44_zernike")
  (defparameter *code-file* "run_00_one_shot")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  (progn
 (defun choose (n k)
   (labels ((prod-enum (s e)
	      (do ((i s (1+ i)) (r 1 (* i r))) ((> i e) r)))
	    (fact (n) (prod-enum 1 n)))
     (/ (prod-enum (- (1+ n) k) n) (fact k))))


 (defun zernike-radial-coef (m n)
   "returns the scale of the term rho**(n-2k)"
   (when (oddp (- n m))
     (break "odd number not supported"))
   (loop for k
	 from 0
	 upto (/ (- n m)
		 2)
	 collect
	 (* 
	  (expt -1 k)
	  (choose (- n k)
		  k)
	  (choose (- n (* 2 k))
		  (- (/ (- n m)
			2)
		     k))))))
     
  (let* (
	 
	 (code
	  `(do0
	    (do0 
		 #-nil(do0
		  
		  (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		  (imports ((plt matplotlib.pyplot)
			  ;  (animation matplotlib.animation) 
                            ;(xrp xarray.plot)
			    ))
                  
		  (plt.ion)
					;(plt.ioff)
		  ;;(setf font (dict ((string size) (string 6))))
		  ;; (matplotlib.rc (string "font") **font)
		  )
	         (imports (		;os
					;sys
					;time
					;docopt
					;pathlib
					;(np numpy)
					;serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;   scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
			   (np numpy)
			   ;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
			  ; copy
			   
			   ))
		
		 (do0
		  (setf coef (list ,@()))
		)
		))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))




(defun osa-index (n l)
  (/ (+ (* n (+ n 2))
	l)
     2))

(defun all-positions (needle haystack &key (key #'identity))
  (loop for e in haystack and position from 0
	when (eql (funcall key e) needle)
	  collect position))

(let* ((mapping
       (sort 
	(remove-if #'null
		   (loop for n from 0 upto 5
			 appending
			 (loop for l from -5 upto 5
			       collect
			       (let ((j (osa-index n l)))
				 (when (and (<= 0 j)
					    (integerp j))
				   `(,j :n ,n  :l ,l))))))
	#'< :key #'first)
       )
       (keys (mapcar #'first mapping))
       (unique-keys (remove-duplicates keys))
       (repeated-keys (remove-if #'null (loop for e in unique-keys
					      collect
					      (unless (<=
						       (count e keys) 1)
						e)))))
  (loop for e in repeated-keys
	collect
	(let ((repeated-positions 
		(all-positions e mapping :key #'first)))
	  (loop for duplicated-index-map in (mapcar #'(lambda (pos) (elt mapping pos))
						    repeated-positions)
		collect
		(destructuring-bind (j &key n l) duplicated-index-map
		  (assert (eq j e))
		  `(j ,e n ,n l ,l n+l ,(+ n l))))))
  )
