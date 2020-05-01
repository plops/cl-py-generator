(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)


(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/15_slab_waveguide_modes")
  (defparameter *code-file* "run_00_slab_mode")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))

  (defun show (vars)
    `(do0
      (print (dot (string ,(format nil "~{~a={}~^ ~}"
				   vars))
		  (format ,@vars)))))
  
  (let* ((code
	  `(do0
	    "# "
	    "# export LANG=en_US.utf8"
	    
	    
	    (do0
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
					;serial
					;pathlib
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					;scipy.ndimage
					;scipy.optimize
		      scipy.sparse.linalg
		      time
		      ))
	    (do0
	     (comment "simulation parameters")
	     (setf lam0 1s0
		   n1 2s0
		   n2 1s0
		   a (* 3 lam0)	 ;; core thickness
		   b (* 5 lam0)	 ;; substrate thickness
		   dx (/ lam0 20) ;; grid resolution
		   M 5		  ;; number of modes to calculate
		   )
	     ,(show `(lam0 n1 n2 a b dx M)))
	    (do0
	     (comment "compute grid")
	     (setf Sx (+ a (* 2 b))
		   Nx (int (np.ceil (/ Sx dx)))
		   Sx (* Nx dx)
		   xa (* dx (np.linspace .5 (- Nx .5) (+ Nx -1)))
		   xa (- xa (np.mean xa)))
	     (comment "start and stop indices (centered in grid)")
	     (setf nx (int (np.round (/ a dx)))
		    
		   nx1 (int (np.round (/ (- Nx nx) 2))) ;; start
		   nx2 (+ nx1 nx -1) ;; end
		   )
	     (setf N (np.ones Nx ;(tuple Nx 1)
			      )
		   (aref N "0:nx1-2") n2
		   (aref N "nx1:nx2") n1
		   (aref N "nx2+1:Nx") n2)
	     ,(show `(Sx Nx ;xa
			 nx nx1 nx2
			 )))
	    (do0
	     (comment "perform fd analysis")
	     (setf k0 (/ (* 2 np.pi) lam0))
	     (setf DX2 (*
			(/ 1s0
			   (** (* k0 dx) 2))
			(scipy.sparse.diags
			 (tuple
			  (* +1 (np.ones (- Nx 1) ))
			  (* -2 (np.ones (- Nx 0) ))
			  (* +1 (np.ones (- Nx 1) )))
			 (tuple -1 0 1)))
		   )
					; (print (DX2.toarray))
	     (setf N2 (scipy.sparse.diags (tuple (** N 2))
					  (tuple 0)))
	     (setf A (+ DX2 N2))
	     (do0
	      (setf Afull (A.toarray))
	      (do0
	       (setf start (time.clock))
	       (setf (tuple D V) (np.linalg.eig Afull))
	       (setf end (time.clock)
		     duration_full (- end start))
	       ,(show `(duration_full)))
	      
	      (setf NEFF (np.real (np.sqrt (+ 0j D)))
		    ))
	     ;; np.real around sqrt?
	     #-nil (do0
		    (do0
	       (setf start (time.clock))
	       (setf (tuple Ds Vs) (scipy.sparse.linalg.eigs A M :which (string "LR")))
	       (setf end (time.clock)
		     duration_sparse (- end start))
	       ,(show `(duration_sparse)))
		    
	      (setf NEFFs (np.sqrt Ds)))
	     )

	    (do0
	     (comment "plot")
	     (setf ind (np.flip (np.argsort NEFF))
		   NEFF1 (np.flip (np.sort NEFF)))
	     (setf V1 (aref V ":" ind))
	     (do0
	      (comment "substrate")
	      (plt.axhline :y (- -b (/ a 2)))
	      (plt.axhline :y (+ b (/ a 2))))
	     (do0
	      (comment "core")
	      (plt.axhline :y (/ -a 2))
	      (plt.axhline :y (/ a 2)))
	     (for (m (range M))
		  (setf x0 (* 2 m)
			y0 (* .5 (+ a b))
			x (+ x0 (* 3 (aref V1 ":" m)))
			xs (+ x0 (* 3 (aref Vs ":" m)))
			y (np.linspace (- -b (/ a 2))
				       (+ b (/ a 2))
				       Nx)
			)
		  (plt.plot x y)
		  (plt.plot xs y)
		  (plt.text x0 y0 (dot (string "mode={}\\n{:6.4f}\\n{:6.4f}")
				       (format m
					       (aref NEFF1 m)
					       (aref NEFFs m)))))
	     )
	    
	    ))) 
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
 

 
    
