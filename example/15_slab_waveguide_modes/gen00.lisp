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
	      (setf (tuple D V) (np.linalg.eig (A.toarray)))
	      (setf NEFF (np.real (np.sqrt (+ 0j D)))
		    ))
	     ;; np.real around sqrt?
	     #+nil (do0
	      (setf (tuple V D) (scipy.sparse.linalg.eigs A))
	      (setf NEFF (np.sqrt (np.diag D))))
	     )

	    (do0
	     (comment "plot")
	     (setf ind (np.flip (np.argsort NEFF))
		   NEFF1 (np.flip (np.sort NEFF)))
	     (setf V1 (aref V ":" ind))
	     (for (m (range M))
		  (plt.plot (aref V1 ":" m)))
	     )
	    
	    ))) 
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))
 

 
    
