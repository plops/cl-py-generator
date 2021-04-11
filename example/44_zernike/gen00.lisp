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

    (progn (defun osa-index-nl-to-j (n l)
   (/ (+ (* n (+ n 2))
	 l)
      2))

       (defun all-positions (needle haystack &key (key #'identity))
	 "find multiple occurances of needle in haystack"
	 (loop for e in haystack and position from 0
	       when (eql (funcall key e) needle)
		 collect position))

       (defun osa-indices-j-to-nl (&key (n 5) (l 5))
	 "Given a 2d zernike index n,l return list of mappings between j to pairs n,l: []"
	 (let* ((mapping ;; [<j> :n <n> :l <l>]* 
		  (sort 
		   (remove-if #'null
			      (loop for n from 0 upto n
				    appending
				    (loop for l from (- l) upto l
					  collect
					  (let ((j (osa-index n l)))
					    (when (and (<= 0 j)
						       (integerp j))
					      `(,j :n ,n  :l ,l))))))
		   #'< :key #'first))
		(keys (mapcar #'first mapping)) ;; [<j>]* with duplicates
		(unique-keys (remove-duplicates keys)) ; [<j>]* without duplicates
		(repeated-keys			; [<j>]* only duplicates
		  (remove-if #'null (loop for e in unique-keys
					  collect
					  (unless (<=
						   (count e keys) 1)
					    e))))
		(single-keys			; [<j>]* that occur only once
		  (remove-if #'null (loop for e in unique-keys
					  collect
					  (when (=
						 (count e keys) 1)
					    e))))
		(mapping-merit ;; [<j> :n <n> :l <l> :merit <abs(n+l)> :merit <abs(n)+abs(l)>]* each j occurs once (merit minimized) 
		  (loop for e in repeated-keys
			collect
			(let ((repeated-positions ; [<pos>]* positions where e is equal to an entry in mapping
				(all-positions e mapping :key #'first)))
			  (let ((all-merit-for-e ;; [<j=e> :n <n> :l <l> :merit <abs(n+l)>]* 
				  (loop for duplicated-index-map ;; acces mapping
					  in (mapcar #'(lambda (pos) (elt mapping pos))
						     repeated-positions)
					collect
					(destructuring-bind (j &key n l) duplicated-index-map
					  (assert (eq j e))
					  ;; compute merit function abs(n+l)
					  `(,e :n ,n :l ,l
					       :merit ,(abs (+ n l))
					       :merit2 ,(+ (abs n) (abs l)))))))
			    ;; sort by merit and merit2 to make results match the example table in wikipedia
			    ;; => (((0 :n 0 :l 0 :merit 0 :merit2 0) (0 :n 1 :l -3 :merit 2 :merit2 4))
			    ;;     ((1 :n 1 :l -1 :merit 0 :merit2 2) (1 :n 0 :l 2 :merit 2 :merit2 2))
			    ;;     ((2 :n 1 :l 1 :merit 2 :merit2 2) (2 :n 0 :l 4 :merit 4 :merit2 4)
			    ;;      (2 :n 2 :l -4 :merit 2 :merit2 6))
			    ;;     ((3 :n 2 :l -2 :merit 0 :merit2 4) (3 :n 1 :l 3 :merit 4 :merit2 4))
			    ;;     ((4 :n 2 :l 0 :merit 2 :merit2 2) (4 :n 1 :l 5 :merit 6 :merit2 6))
			    ;;     ((5 :n 2 :l 2 :merit 4 :merit2 4) (5 :n 3 :l -5 :merit 2 :merit2 8))
			    ;;     ((6 :n 3 :l -3 :merit 0 :merit2 6) (6 :n 2 :l 4 :merit 6 :merit2 6))
			    ;;     ((10 :n 4 :l -4 :merit 0 :merit2 8) (10 :n 3 :l 5 :merit 8 :merit2 8)))
			    ;;
			    ;; only return the first element (with best merit functions)
			    (first
			     (stable-sort
			      (sort all-merit-for-e #'< :key #'(lambda (x)
								 (destructuring-bind (j &key n l merit merit2) x
								   merit)))
			      #'< :key #'(lambda (x)
					   (destructuring-bind (j &key n l merit merit2) x
					     merit2)))))))))

	   (sort ;; merge unique j->(n,l) mappings and the best mapping of j->(n,l) mappings with mulitplicity
	    (append
	     mapping-merit
	     (loop for e in single-keys
		   collect (find e mapping :key #'first)))
	    #'< :key #'first)))

       (defun choose (n k)
	 "compute binomial coefficent"
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
			   k)))))))
     
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





