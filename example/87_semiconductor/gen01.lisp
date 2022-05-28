(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

;; pip3 install --user plotly cufflinks chart_studio pycairo diplib
;; pip3 install --user plotly --upgrade

;; Plotly Tutorial 2021 https://www.youtube.com/watch?v=GGL6U0k8WYA

(progn
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/87_semiconductor")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key (msg "") vars)
    `(print (dot (string ,(format nil "{:7.6f} \\033[31m ~a \\033[0m ~{~a={}~^ ~}" msg vars))
                 (format
		  (- (time.time) start_time)
                  ,@vars))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ()
	(gen `(locate)
	     `((python
		(cell
		 #+nil
		 (do0 (imports ((plt matplotlib.pyplot)))
		      (plt.ion))
		 (imports (pathlib
			   time
			   (pd pandas)
			   (np numpy)
			   cairo
			   (cf cufflinks)
			   (py chart_studio.plotly)
			   (px plotly.express)
			   (go plotly.graph_objects)
			   scipy.signal
			   (dip diplib)
			   tqdm
					;(o3d open3d)
			   ))
		 "%matplotlib inline"
		 #+nil (imports-from (matplotlib.pyplot plot figure scatter gca sca subplots subplots_adjust title xlabel ylabel xlim ylim grid))
		 (imports-from (plotly.offline download_plotlyjs
					       init_notebook_mode
					       plot
					       iplot))
		 (init_notebook_mode :connected True)
		 (cf.go_offline)
		 (setf start_time (time.time))))
	       #+nil
	       (python
		(cell
		 (setf df (pd.DataFrame
			   (np.random.randn 50 4)
			   :columns (list (string "A")
					  (string "B")
					  (string "C")
					  (string "D"))))
		 (df.iplot)
		 ))

	       (python
		(cell
		 (comments "render a disk (mccutchen pupil)")
		 (setf nx 171
		       ny 233
		       (ntuple xx yy)
		       (dot np (aref mgrid
				     (slice "" nx)
				     (slice "" ny)))
		       rr (+ (** (- xx
				    (* .5 nx))
				 2)
			     (** (- yy
				    (* .5 ny))
				 2))
		       disk (< rr (** (* .5 70)
				      2)))
		 (px.imshow disk)))
	       (python
		(cell
		 (setf kdisk (np.fft.fft2 (* disk 1s0))
		       kdisk_centered (np.fft.fftshift
				       kdisk
					;:axes (tuple -2)
				       ))
		 (px.imshow (np.log (np.abs kdisk_centered))
			    :aspect (string "equal"))))
	       (python
		(cell
		 (comments "psf")
		 (setf psf
		       (* kdisk
			  (np.conj kdisk)))
		 (setf psf_view (np.fft.fftshift (np.abs psf)
					;:axes (tuple -2)
						 )
		       psf_view (/ psf_view (np.sum psf_view)))
		 (px.imshow psf_view)

		 ))
	       (python
		(cell
		 (comments "show cross-section of psf, 4px approximately 250nm; 1px ~ 62.5nm")
		 (px.line (aref psf_view
				(// nx 2)
				(slice 100 130)))))
	       (python
		(cell
		 (do0
		  (setf fig (go.Figure :data (list (go.Surface
						    :z psf_view))))
		  (fig.update_layout :title (string "psf")
				     :width 500 :height 500)
		  (fig.show))))
	       (python
		(cell
		 (comments "compute modulation transfer function")
		 (setf mtf (np.real (np.fft.fftshift
				     (np.fft.ifft2 psf
						   :s disk.shape))))
		 (setf mtf (/ mtf (np.max mtf)))
		 (px.imshow mtf)))
	       (python
		(cell
		 (comments "render mtf as 3d surface")
		 (do0
		  (setf fig (go.Figure :data (list (go.Surface
						    :x xx
						    :y yy
						    :z mtf))))
		  (fig.update_layout :title (string "mtf")
				     :scene
				     (dictionary :xaxis_title (string "kx")
						 :yaxis_title (string "ky"))
				     :width 500 :height 500)
		  (fig.show))
		 ))
	       (python
		(cell
		 (comments "draw marker")
		 (comments "https://stackoverflow.com/questions/10031580/how-to-write-simple-geometric-shapes-into-numpy-arrays")
		 (comments "looks like x and y are swapped in cairo")
		 (setf data (np.zeros (tuple nx ny 4)
				      :dtype np.uint8)
		       surface (cairo.ImageSurface.create_for_data
				data
				cairo.FORMAT_ARGB32 ny nx)
		       cr (cairo.Context surface)
		       )

		 (do0
		  (comments "clear with black")
		  (cr.set_source_rgb 0s0 0s0 0s0)
		  (cr.paint))
		 (do0
		  (comments "https://pycairo.readthedocs.io/en/latest/reference/context.html")
		  ;; xc yc radius angle1 angle2
		  (setf ang1 (/ (* 1 2 np.pi) 6s0)
			ang2 (+ (* 120 (/ (* 2 np.pi)
					  180))
				(/ (* -1 2 np.pi) 6s0)))
		  (cr.arc  (* .5 ny)
			   (* .5 nx)
			   30
			   ang1
			   ang2
			   )
		  (cr.set_line_width 5)
		  (cr.set_source_rgb 1s0 1s0 1s0)
		  (cr.stroke)
		  (do0
		   (cr.set_line_width 3)
		   (setf rad2 40
			 n_spokes 3
			 )
		   (for (spoke (range n_spokes))
			(setf ang (+ .2 (* 2 np.pi (/ -1d0 6))
				     (/ (* 2 np.pi spoke)
					n_spokes)))

			(do0 ;; x y
			 (cr.move_to (* .5 ny)
				     (* .5 nx))
			 (cr.line_to (+ (* .5 ny) (* rad2 (np.cos ang)))
				     (+ (* .5 nx) (* rad2 (np.sin ang))))
			 (cr.stroke)))))
		 (setf marker (aref data ":" ":" 0))
		 (px.imshow marker)
		 ))

	       (python
		(cell
		 (comments "blur marker with psf")
		 (setf img (scipy.signal.fftconvolve
			    psf_view
			    marker
			    :mode (string "same")))
		 (setf img (/ img (np.max img)))
		 (px.imshow img)))
	       (python
		(cell
		 (comments "simulate poisson noise")
		 (setf max_photons_per_pixel 10)
		 (setf rng (np.random.default_rng)
		       img_pois (rng.poisson :lam (* max_photons_per_pixel
						     img)))
		 (px.imshow img_pois)))
	       (python
		(cell
		 (comments "use diplib to find shift between blurred marker and noisy image")
		 (comments "https://qiftp.tudelft.nl/dipref/FindShift.html")
		 (setf (ntuple dx dy) (dip.FindShift img img_pois))
		 dx
		 ))
	       (python
		(cell
		 (comments "perform a few repeats with different number of photons")
		 (setf res (list))
		 (setf nm_per_px 62.5)
		 (for (max_phot (list 10 20 30 100 1000 10000))
		      (for (rep (tqdm.tqdm (range 100)))
			   (setf
			    img_pois (rng.poisson :lam (* max_phot
							  img)))
			   (comments "normalize image to same values as input")
			   (setf (ntuple dx dy) (dip.FindShift img (/ img_pois
								      (np.max img_pois))))
			   (res.append
			    (dictionary :max_phot max_phot
					:rep rep
					:dx (* nm_per_px dx)
					:dy (* nm_per_px dy)))))
		 (setf df (pd.DataFrame res))
		 (df.to_csv (string "/home/martin/stage/cl-py-generator/example/87_semiconductor/source/dir87_gen01_location.csv")))
		)
	       (python
		(cell
		 (setf fig (px.histogram df :x (string "dx")
					 :marginal (string "violin")
					 :color (string "max_phot")))
		 (fig.update_layout :xaxis_title_text (string "x shift estimate (nm)"))))
	       (python
		(cell
		 (setf fig (px.histogram df :x (string "dy")
					 :marginal (string "violin")
					 :color (string "max_phot")))
		 (fig.update_layout :xaxis_title_text (string "y shift estimate (nm)")))))))))
  #+nil (sb-ext:run-program "/usr/bin/sh"
			    `("/home/martin/stage/cl-py-generator/example/87_semiconductor/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




