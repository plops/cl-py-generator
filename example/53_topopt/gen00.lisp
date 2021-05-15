(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/53_topopt")
  (defparameter *code-file* "run_00_topopt")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
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
			   scipy.sparse
			   scipy.sparse.linalg
					; jax
			   ;jax.random
					;jax.config
			   ;copy
			   ;subprocess
			   datetime
			   time
			   ))
		
		 (setf
		  _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

		  _code_generation_time
		  (string ,(multiple-value-bind
				 (second minute hour date month year day-of-week dst-p tz)
			       (get-decoded-time)
			     (declare (ignorable dst-p))
			     (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				     hour
				     minute
				     second
				     (nth day-of-week *day-names*)
				     year
				     month
				     date
				     (- tz))
			     )))
		 (do0
		  (setf nelx 180
			nely 60
			volfrac .4
			rmin 5.4
			penal 3.0
			ft 1 ;; filter method: 0 .. sensitivity, 1 .. density based  
			
			)
		  (def lk ()
		    (setf E 1
			  nu .3
			  k (np.array (list (- .5 (/ nu 6))
					    (+ ,(* 1.0 1/8) (/ nu 8))
					    (- ,(* 1.0 -1/4) (/ nu 12))
					    (+ ,(* 1.0 -1/8) (* 3 (/ nu 8)))
					    (+ ,(* 1.0 -1/4) (/ nu 12))
					    (- ,(* 1.0 -1/8) (/ nu 8))
					    (/ nu 6)
					    (- "1/8" (* 3 (/ nu 8)))))
			  KE (* (/ E (- 1 (** nu 2)))
				(np.array
				 (list
				  ,@(loop for e in `(0 1 2 3 4 5 6 7
						       1 0 7 6 5 4 3 2
						       2 7 0 5 6 3 4 1
						       3 6 5 0 7 2 1 4
						       4 5 6 7 0 1 2 3
						       5 4 3 2 1 0 7 6
						       6 3 4 1 2 7 0 5
						       7 2 1 4 3 6 5 0)
					  collect
					  `(aref k ,e))))))
		    (return KE))
		  ;; optimality criterion
		  (def oc (&key nelx nely x volfrac dc dv g)
		    (setf l1 0
			  l2 1e9
			  move .2
			  xnew (np.zeros (* nelx nely)))
		    (while (< 1e-3 (/ (- l2 l1)
				      (+ l1 l2)))
			   (setf lmid (* .5 (+ l2 l1))
				 (aref xnew ":")
				 (np.maximum 0.0
					     (np.maximum (- x move)
							 (np.minimum 1.0
								     (np.minimum (+ x move)
										 (* x (np.sqrt (/ -dc (* dv lmid))))))))
				 gt (+ g (np.sum (* dv (- xnew x)))))
			   )
		    (if (< 0 gt)
			       (setf l1 lmid)
			       (setf l2 lmid))
		    (return (tuple xnew gt)))
		  (do0
		   (setf Emin 1e-9
			 Emax 1.0
			 ndof (* 2 (+ nelx 1)
				 (+ nely 1))
			 x (* volfrac (np.ones (* nely nelx)
					       :dtype float))
			 xold (x.copy)
			 xPhys (x.copy)
			 g 0
			 dc (np.zeros (list nely nelx)
				      :dtype float)
			 KE (lk)
			 edofMat (np.zeros (list (* nelx nely) 8)
					   :dtype int)
			 )
		   (for (elx (range nelx))
			(for (ely (range nely))
			     (setf el (+ ely (* elx nely))
				   n1 (+ (* (+ nely 1)
					    elx)
					 ely)
				   n2 (+ (* (+ nely 1)
					    (+ elx 1))
					 ely)
				   (aref edofMat el ":")
				   (np.array (list ,@(loop for (e f) in `((n1 2)
									  (n1 3)
									  (n2 2)
									  (n2 3)
									  (n2 0)
									  (n2 1)
									  (n1 0)
									  (n1 1))
							  collect
							   `(+ (* 2 ,e) ,f)))))))
		   (setf iK (dot (np.kron edofMat (np.ones (tuple 8 1)))
				 (flatten)))
		   (setf jK (dot (np.kron edofMat (np.ones (tuple 1 8)))
				 (flatten)))
		   (setf nfilter (int (* nelx
					 nely
					 (** (+ 1
						(* 2
						 (- (np.ceil rmin)
						    1)))
					     2))))
		   ,@(loop for e in `(iH jH sH)
			   collect
			   `(setf ,e (np.zeros nfilter)))
		   (setf cc 0)
		   (for (i (range nelx))
			(for (j (range nely))
			     (setf row (+ (* i nely) j)
				   
				   )
			     (setf 
				    kk1 (int (np.maximum 0
							 (- i (- (np.ceil rmin) 1))))
				    kk2 (int (np.maximum nelx
							 (+ i (np.ceil rmin)))))
			     (setf 
				    ll1 (int (np.maximum 0
							 (- j (- (np.ceil rmin) 1))))
				    ll2 (int (np.maximum nely
							 (+ j (np.ceil rmin)))))
			     (for (k (range kk1 kk2))
				  (for (l (range ll1 ll2))
				       (setf col (+ l (* k nely))
					     fac (- rmin
						    (np.hypot (- i k)
							      (- j l))))
				       (setf (aref iH cc) row
					     (aref jH cc) col)
				       (setf (aref sH cc) (np.maximum 0.0
								      fac))
				       (setf cc (+ cc 1))))
			     )))
		  )))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



