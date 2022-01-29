(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)


(progn
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/78_django")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defparameter *libs*
    `(;(np numpy)
      ;(cv cv2)
      ;(pd pandas)
      ;(xr xarray)
      ;lmfit
      ;rawpy
      ;wx
      ))
  (defun scale (da &key (perc 2) (range 10))
    (let ((d `(- ma mi)))
      `(do0
	(setf mi (np.nanpercentile ,da ,perc)
	      ma (np.nanpercentile ,da (- 100
					 ,perc))
	      mi1 (- mi (* ,(/ range 100.0) ,d))
	      ma1 (+ ma (* ,(/ range 100.0) ,d))
	      ))))
  (defun lprint (&optional rest)
    `(print (dot (string ,(format nil "~{~a={}~^\\n~}" rest))
                 (format  ;(- (time.time) start_time)
                  ,@rest))))
  (flet ((out (fn lines)
	   (with-open-file (s (format nil "~a/source/web/~a" *path* fn)
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
	(format s "~{~a~^~%~}" lines))))
    (out "requirements.txt"
	 `(Django
	   Pillow))
    (out "setup01_create_venv.sh"
	 `("# create virtual environment"
			     "python3 -m venv new env"
			     "source env/bin/activate"))
    (out "setup02.sh"
	 `("# install dependencies into virtual environment"
			     "pip install -r requirements.txt"
			     "# undo: rm -rf env")
	 )
    (out "setup03.sh"
	 `("# bootstrap django project"
			     "django-admin startproject pygram ."
			     "# undo: rm -rf pygram manage.py")
	 )
    (out "setup04_runserver.sh"
	 `("# start django webserver"
	   "./manage.py runserver")
	 )
    (out "setup05_startapp_posts.sh"
	 `("# add django application with business logic for creating andd commentiiing on posts"
	   "./manage.py startapp posts"
	   "# undo: rm -rf posts")
	 ))
  
  
  (let ((nb-file "source/01_posts_models.ipynb"))
   (write-notebook
    :nb-file (format nil "~a/~a" *path* nb-file)
    :nb-code
    `((python (do0
	       "# default_exp posts/models"
	       ))
      (python (do0
	       "#export"
	       (do0
		(imports-from (django.db models)))
	       ))
      
      (python
       (do0
	"#export"
	(class Post (models.Model)
	      (setf image (models.ImageField)))))
      ))))



 
