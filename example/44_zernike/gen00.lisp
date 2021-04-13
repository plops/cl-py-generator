(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/44_zernike")
  (defparameter *code-file* "run_00_zernike")
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

    (progn
      (defun osa-index-nl-to-j (n l)
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
					  (let ((j (osa-index-nl-to-j n l)))
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

       
       (defun zernike-radial-coef (n l)
	 "returns the scale of the term rho**(n-2k)"
	 (when (oddp (- n l))
	   (break "odd number not supported"))
	 (loop for k
	       from 0
	       upto (/ (- n l)
		       2)
	       collect
	       (list
		:degree (- n (* 2 k))
		:coef
		(* 
		 (expt -1 k)
		 (choose (- n k)
			 k)
		 (choose (- n (* 2 k))
			 (- (/ (- n l)
			       2)
			    k)))
		:n n
		:l l
		:k k)))
       (defun zernike-radial-coef-sorted (n l)
	 "return coefficients of the radial polynomial sorted by degree (lowest degree first)"
	 (sort 
	  (zernike-radial-coef n l)
	  #'<
	  :key #'(lambda (x) (destructuring-bind (&key degree coef n l k) x
			       degree))))))
  
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
                            (xrp xarray.plot)
			    ))
                  
		  (plt.ion)
					;(plt.ioff)
		  (setf font (dict ((string size) (string 6))))
		  (matplotlib.rc (string "font") **font)
		  )
	         (imports (		;os
					;sys
					;time
					;docopt
					;pathlib
					;(np numpy)
					;serial
			   (pd pandas)
			   (xr xarray)
			   (xrp xarray.plot)
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
		 ,(let* ((lmax 10)
			 (nmax 10)
			 (lut-indices (osa-indices-j-to-nl :n nmax :l lmax))
			 #+nil (jmax (loop for e in lut-indices
				     maximize
				     (destructuring-bind 
					 (j &key n l merit merit2) e
				       j)
				     )))
					;(declare (ignorable jmax))
		    (defparameter *lut* lut-indices)
		    `(do0
		      (def osa_index_nl_to_j (n l)
			(return (// (+ (* n (+ n 2))
				      l)
				   2)))
		      (def osa_index_j_to_nl (j)
			(setf lut (list ,@(loop for e in lut-indices
						collect
						(destructuring-bind 
						    (j &key n l merit merit2) e
						  `(list ,n ,l)))))
			(return (aref lut j)))
		      (def zernike (rho phi &key (n 0) (l 0))
			(comments ,(format nil "n in [0 .. ~a], l in [-~a .. ~a]" nmax lmax lmax))
			(setf arg (* phi (abs l)))
			(if (< l 0)
			    (setf azi (np.sin arg))
			    (setf azi (np.cos arg)))
			(comments "polynomial coefficients in order of increasing degree, (1 2 3) is 1 + 2*x + 3*x**2")
			(setf coef
			      (list ,@(loop for e in lut-indices
					    collect
					    (destructuring-bind (j &key n l merit merit2) e
					      
					      `(list ,@(let* ((coef-sparse (zernike-radial-coef-sorted n (abs l)))
							      ;; create a full set of coefficients (for each degree of the polynomial)
							      (n-coef-sparse (length coef-sparse))
							      (max-degree (destructuring-bind (&key degree coef n l k) (car (last coef-sparse))
									    degree))
							      (coef-full (loop for c from 0 upto max-degree collect 0)))
							 
							 #+nil (defparameter *sparse* (list :coef-sparse coef-sparse
										      :max-degree max-degree))
							 (loop for c in coef-sparse
							       do
								  (destructuring-bind (&key degree coef n l k) c
								     #+nil (progn (format t "degree=~a coef-full,coef=~a nlk=~a~%" degree (list coef-full coef) (list n l k))
									   (defparameter *bla* (list :max-degree max-degree
												     :n-sparse n-coef-sparse
												     :sparse coef-sparse
												     :coef-full coef-full
												     :n-full (length coef-full
														     )
												     :nlk (list n l k)
												     :degree degree)))
								     (setf (elt coef-full degree) coef)
								    ))
							 coef-full)))
						 )))
			(setf osa_index (osa_index_nl_to_j n l))
			
			(setf radial (np.polynomial.polynomial.polyval rho (aref coef osa_index)))
			(setf mask (np.where (< rho 1) 1s0 np.nan))
			(return (* mask radial azi))
			)
		      (do0
		       (do0 (setf x (np.linspace -1 1 128)
			      y (np.linspace -1 1 128)
			      rho (np.hypot x (aref y ":" np.newaxis))
			      phi (np.arctan2 (aref y ":" np.newaxis) x))
			    (setf zval (zernike rho phi :n 1 :l 1))
			    
			    (setf xs (xr.DataArray :data zval
						   :coords (list y x)
						   :dims (list (string "y")
							       (string "x")))))
		       #+nil (do0 (xs.plot)
			    (setf cs (xrp.contour xs  :colors (string "k")))
			    (plt.clabel cs :inline True)
			    (plt.grid))
		       )
		      (do0
		       (def xr_zernike (&key (n 0) (l 0) (x (np.linspace -1 1 64))
					     (y (np.linspace -1 1 64)))
			 (string3 "return xarray with evaluated zernike polynomial")
			 (do0 (setf
			      rho (np.hypot x (aref y ":" np.newaxis))
			      phi (np.arctan2 (aref y ":" np.newaxis) x))
			    (setf zval (zernike rho phi :n n :l l))
			    
			    (setf xs (xr.DataArray :data zval
						   :coords (list y x)
						   :dims (list (string "y")
							       (string "x"))))
			    (return xs))))
		      ,(let ((l `((1 1 tilt)
				  (2 0 defocus)
				  (2 2 primary-astigmatism)
				  (3 1 primary-coma)
				  (3 3 trefoil)
				  (4 0 primary-spherical)
				  (4 2 secondary-astigmatism)
				  (4 4 tetrafoil)
				  (5 1 secondary-coma)
				  (5 3 secondary-trefoil)
				  (5 5 pentafoil)
				  (6 0 secondary-spherical)
				  (6 2 tertiary-astigmatism)
				  (6 4 secondary-trefoil)
				  (6 6 hexafoil)
				  (7 1 tertiary-coma)
				  (7 3 tertiary-trefoil)
				  (7 5 secondary-pentafoil)
				  (7 7 heptafoil)
				  (8 0 tretiary-spherical))
				))
			`(do0
			 (plt.figure :figsize (list 19 10))
			 (setf zernike_names (pd.DataFrame
					      (dict ((string "n") (list ,@(mapcar #'first l)))
						    ((string "l") (list ,@(mapcar #'second l)))
						    ((string "name") (list ,@(mapcar #'(lambda (x) `(string ,(third x))) l))))
					      ))
			 (for (j (range 0 (* 4 9)))
			      (do0
			       (setf ax (plt.subplot 4 9 (+ j 1)))
			       (ax.set_aspect (string "equal"))
			       (setf (tuple n l) (osa_index_j_to_nl j))
			       (setf xs (xr_zernike n l))
			       (do0 (xs.plot :vmin -1 :vmax 1 :add_colorbar False)
				    (setf cs (xrp.contour xs  :colors (string "k")))
				    (plt.clabel cs :inline True)
				    (plt.grid)
				    (setf lookup (aref zernike_names (& (== zernike_names.n n)
									(== zernike_names.l (np.abs l)))))
				    (if (== 1 (len lookup))
				     (plt.title (dot (string "j={} n={} l={}\\n{}")
						     (format j n l
							     (dot lookup 
							      name
							      (item)))))
				     (plt.title (dot (string "j={} n={} l={}")
						     (format j n l)))))
			       ))
			 (plt.tight_layout :rect (list 0 0 1 .98))
			 (plt.savefig (string "zernikes.png"))))))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



