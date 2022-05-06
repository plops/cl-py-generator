(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)
;; http://www.open3d.org
;; sudo dnf install python3.9
;; python3.9 -m ensurepip
;; python3.9 -m pip install --user open3d ;; does not work with python 3.10
;; http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
;; http://www.open3d.org/docs/0.9.0/tutorial/Basic/pointcloud.html
(progn
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/85_o3d_pointcloud")
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
	(gen `(cloud)
	     `((python
		(cell
		 (imports ((plt matplotlib.pyplot)))
		 (plt.ion)
		 (imports (pathlib
			   time
			   (pd pandas)
			   (np numpy)
					;tqdm
			   (o3d open3d)
			   ))
		 (imports-from (matplotlib.pyplot plot figure scatter gca sca subplots subplots_adjust title xlabel ylabel xlim ylim grid))
		 (setf start_time (time.time))))

	       (python
		(cell
		 (setf fn (str (next (dot pathlib
					  (Path (string "/home/martin/t5/3d/"))
					  (glob (string "*.ply")))
				     )))
		 ,(lprint :vars `(fn))
		 (setf pcd (dot o3d
				io
				(read_point_cloud
				 fn
				 ;; :format (string "xyzrgb")
				 )))
		 (o3d.visualization.draw_geometries (list pcd))
		 ))
	       )))))
  (sb-ext:run-program "/usr/bin/sh"
		      `("/home/martin/stage/cl-py-generator/example/85_o3d_pointcloud/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




