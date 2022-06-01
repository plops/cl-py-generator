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
  (defparameter *example-subdir* "example/88_plotly")
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
	(gen `(plot)
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
			   ;scipy.signal
			   ;(dip diplib)
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
	       (python
		(cell
		 (setf df_ (pd.read_csv (string "chart.csv")
				       :delimiter (string ";")))))
	       (python
		(cell
		 (setf rename_dict (dict
				 ,@(loop for (e f) in
					 `((time "time")
					   (rpm "Engine speed: ---")
					   (valve "Throttle valve position (absolute): ---")
					   (cool "Coolant temperature: ---")
					   (airflowrat "Standardised airflow ratio: ---")
					   (airmass "Air mass, actual value:: ---")
					   (torque "Engine torque: ---")
					   (valve1 "Throttle valve position (absolute): ---.1")
					   (aircleaner "Air cleaner degree of soiling: ---")
					   (airpressure "Charge air pressure, actual value: ---")
					   (fuelpressure "Fuel Pressure: ---")
					   (exhaustmass "Exhaust mass flow of EGR: ---"))
					 collect
					 `((string ,f) (string ,e)))))
		 (setf df (dot df_
			       (rename
				:columns
				rename_dict)))))
	       ,@(loop for e in `((:name rpm :mi 800 :ma 2000) 
				  (:name valve)
				  (:name cool)
				  (:name airflowrat :mi 1 :ma 2)
				  (:name airmass)
				  (:name torque :mi -5 :ma 200)
				  ;(:name valve1)
				  (:name aircleaner)
				  (:name airpressure)
				  (:name fuelpressure :mi 20e3 :ma 100e3)
				  (:name exhaustmass)
				  )
		       collect
		       (destructuring-bind (&key name mi ma) e
			 `(python
			   (cell
			    (setf fig (px.line df :y (string ,name)))
			    ,(if (and mi ma)
				 `(fig.update_layout :yaxis_range (list ,mi ,ma))
				 `(comments "no extra")))))))))))
  
  #+nil
  (progn (sb-ext:run-program "/usr/bin/sh"
			     `("/home/martin/stage/cl-py-generator/example/87_semiconductor/source/setup01_nbdev.sh"))
	 (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC )))




