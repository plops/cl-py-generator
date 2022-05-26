(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

;; pip3 install --user plotly cufflinks chart_studio pycairo
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
					;tqdm
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
		       psf_view (/ psf_view (np.max psf_view)))
		 (px.imshow psf_view)

		 ))
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
		  (cr.arc  (* .5 nx)
			  (* .5 ny)
			   30
			   0
			   (* (/ 2s0 3)
			      (* 2 np.pi)))
		  (cr.set_line_width 3)
		  (cr.set_source_rgb 1s0 1s0 1s0)
		  (cr.stroke)
		  (do0
		   (setf rad2 40)
		   
		   ,@(let ((n-spokes 3))
		       (loop for spoke below n-spokes
			     collect
			     (let ((ang (+ (/ 1d0 6) (* 2 pi (/ spoke n-spokes)))))
			      `(do0 ;; x y
				(cr.move_to (* .5 nx)
					    (* .5 ny))
				(cr.line_to (+ (* .5 nx) (* rad2 ,(cos ang)))
					    (+ (* .5 ny) (* rad2 ,(sin ang))))
				(cr.stroke)))))))
		 (px.imshow (aref data ":" ":" 0))
		 ))
	       )))))
  #+nil (sb-ext:run-program "/usr/bin/sh"
			    `("/home/martin/stage/cl-py-generator/example/87_semiconductor/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




