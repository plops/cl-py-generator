(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/46_opticspy")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (progn


      (defun fringe-index-nl-to-j (n l)
	(+ (expt (+ 1 (/ (+ n (abs l))
			 2))
		 2)
	  (* -2 (abs l))
	  (* (signum l)
	     (/ (- 1 (signum l))
		2))))


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
			       degree)))))
  (write-notebook
   :nb-file (format nil "~a/source/05_zernike.ipynb" *path*)
   :nb-code
   `(
     (python
      (do0
       "# default_exp ray"))
     (python (do0 "#export"
		  (do0
		   "%matplotlib notebook"
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
			     time
					;docopt
			     pathlib
					;(np numpy)
					;serial
			     (pd pandas)
					(xr xarray)
					(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
			     scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
					; (np numpy)
					;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
					; copy
			     (opr opticspy.ray_tracing)
					;cProfile
			     (np jax.numpy)
			     jax
			     jax.random
			     jax.config
			     IPython
			   
			     ))
		   "from opticspy.ray_tracing.glass_function.refractiveIndex import *"

		   "from jax import grad, jit, jacfwd, jacrev, vmap, lax, random, value_and_grad"
		   ,(format nil "from jax.numpy import ~{~a~^, ~}" `(sqrt newaxis sinc abs))
		   ,(format nil "from jax import ~{~a~^, ~}" `(grad jit jacfwd jacrev vmap lax random))
		 

		 
		 
		   ,(format nil "from matplotlib.pyplot import ~{~a~^, ~}"
			    `(plot imshow tight_layout xlabel ylabel
				   suptitle contour contourf clabel title subplot subplot2grid grid
				   legend figure gcf xlim ylim))

		   (IPython.core.display.display
		    (IPython.core.display.HTML
		     (string "<style>.container { width:100% !important; }</style>")
		     )
		    )
	       
		   (jax.config.update (string "jax_enable_x64")
				      True)
		   (do0
		    (comments "https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions")
		    (class Timer (object)
			   (def __init__ (self &key (name None))
			     (setf self.name name))
			   (def __enter__ (self)
			     (setf self.tstart (time.time)))
			   (def __exit__ (self type value traceback)
			     (print (dot (string "[{}] elapsed: {}s")
					 (format self.name (- (time.time)
							      self.tstart)))))))
		 
		   )
		  ))
     (python
      (do0
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
		      (do0
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
			 (return (aref lut j))))
		      (do0
		       (def fringe_index_nl_to_j (n l)
			 (return (+ (** (+ 1 (/ (+ n (np.abs l))
						2))
					2)
				    (* -2 (np.abs l))
				    (* (np.sign l)
				       (/ (- 1 (np.sign l))
					  2))))))
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
		      ,(let ((l `((0 0 piston)
				  (1 1 tilt)
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
				)
			     (h 4)
			     (w 9))
			`(do0
			 (plt.figure :figsize (list 19 10))
			 (setf zernike_names (pd.DataFrame
					      (dict ((string "n") (list ,@(mapcar #'first l)))
						    ((string "l") (list ,@(mapcar #'second l)))
						    ((string "name") (list ,@(mapcar #'(lambda (x) `(string ,(third x))) l))))
					      ))
			 (for (j (range 0 (* ,h ,w)))
			      (do0
			       (setf ax (plt.subplot ,h ,w (+ j 1)))
			       (ax.set_aspect (string "equal"))
			       (setf (tuple n l) (osa_index_j_to_nl j))
			       (setf xs (xr_zernike n l))
			       #+nil (setf xsm (/ xs (np.maximum (np.nanmax xs)
							   (np.nanmax (* -1 xs)))))
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
						     (format j n l))))
				    (unless (== (% j ,w) 0)
				      (plt.ylabel None))
				    (unless (== (% (// j ,w) ,h) ,(- h 1))
				      (plt.xlabel None)))
			       ))
			 (plt.tight_layout ;:rect (list 0 0 1 .98)
			  )
			 (plt.savefig (string "zernikes.png"))))))))

     )))



