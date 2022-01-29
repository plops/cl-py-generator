(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "spinneret")
  (ql:quickload "lass")
  (ql:quickload "cl-py-generator")
  )

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
  (let ((shell-counter 1))
   (flet ((out (fn lines &key (count t))
	    (with-open-file (s (if count
				   (format nil "~a/source/web/setup~3,'0d_~a" *path* shell-counter fn)
				   (format nil "~a/source/web/~a" *path* fn))
			       :direction :output
			       :if-exists :supersede
			       :if-does-not-exist :create)
	      (format s "~{~a~^~%~}" lines))
	    (when count
	      (incf shell-counter))))
     (out "requirements.txt"
	  `(Django
	    Pillow)
	  :count nil)
     (out "create_venv.sh"
	  `("# create virtual environment"
	    "python3 -m venv new env"
	    "source env/bin/activate"))
     (out "pip_install.sh"
	  `("# install dependencies into virtual environment"
	    "pip install -r requirements.txt"
	    "# undo: rm -rf env")
	  )
     (out "startproject.sh"
	  `("# bootstrap django project"
	    "django-admin startproject pygram ."
	    "# undo: rm -rf pygram manage.py")
	  )
     (out "runserver.sh"
	  `("# start django webserver"
	    "./manage.py runserver")
	  )
     (out "startapp_posts.sh"
	  `("# add django application with business logic for creating andd commentiiing on posts"
	    "./manage.py startapp posts"
	    "# undo: rm -rf posts")
	  )
     (out "makemigrations.sh"
	  `("# tell django to make migrations"
	    "./manage.py makemigrations"
	    "# undo: rm -rf posts/migrations ")
	  )
     (out "migrate.sh"
	  `("# tell django to make migrations"
	    "./manage.py migrate"
	    "# undo: rm -rf posts/migrations ")
	  )
     (out "sqlite_tables.sh"
	  `("# look at the tables in the database"
	    "sqlite3 db.sqlite3 .tables"
	    "# undo: n.a.")
	  )
     (out "manage_createsuperuser.sh"
	  `("# look at the tables in the database"
	    "./manage.py createsuperuser"
	    "echo 'now try to login with your new user at http://127.0.0.1:8000/admin'"
	    "# undo: ")
	  ))) 

   (defmacro with-page ((&key title)
		       &body body)
    `(spinneret:with-html-string
       (:doctype)
       (:html
	(:head
	 (:title ,title)
	 (:style :type "text/css"
		 (lass:compile-and-write
 `(body :font-family "sans-serif"
	)
 `(.container :width 25% :margin auto)
 `(.header :patting 15px
	   :text-align center
	   :font-size 2em
	   :background "#f2f2f2"
	   :margin-bottom 15px)
 #+nil`(.header>a   :color inherit
	       :text-decoration none)
 )))
	(:body ,@body))))
  (let ((nb-counter 1))
   (flet ((gen (path code) 
	    (write-notebook
	     :nb-file (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)
	     :nb-code (append `((python (do0
		     ,(format nil "# default_exp ~{~a~^/~}" path)
		     ))) code))
	    (incf nb-counter))
	  (gen-html (path code)
	    (let ((fn (format nil "~a/source/web/~{~a~^/~}" *path* path)))
	      (ensure-directories-exist (pathname fn))
	     (with-open-file (s fn
				:direction :output
				:if-exists :supersede
				:if-does-not-exist :create)
	       (format s "~a" code)))))
     (gen-html `(posts templates posts base.html)
	       (with-page (:title "PyGram")
		 (:div :class "header"
		       (:a :href (string "/")
			   "PyGram"
			   ))
		 (:div :class "container"
		       "
{% block content %}
"
		       "

{% endblock %}
")))
    #+nil (gen-html `(posts templates posts post_list.html)
	       (with-page (:title "PyGram")
		 "
{% for post in object_list %}
"
		 (:strong "{{ post.author.username }}")
		 
		 (:br)
		 (:img :src "{{ post.image.url }}"
		       :width 400
		       :height 400)
		 (:p (:em "{{ post.created }}")
		     (:br)
		     "{{ post.description }}")
		 
		 "
{% endfor %}
"))
     (gen-html `(posts templates posts post_list.html)
	       (format nil "{% extends 'posts/base.html' %}
{% block content %}
  {% for post in object_list %}
  ~a
  {% endfor %}
{% endblock %}
"
		(spinneret:with-html-string
		  (:strong "{{ post.author.username }}")
		  (:br)
		  (:img :src "{{ post.image.url }}"
			:width 400
			:height 400)
		  (:p (:em "{{ post.created }}")
		      (:br)
		      "{{ post.description }}")
		  )))
     (gen `(posts models)
	  `(
	    (python (do0
		     "#export"
	       (do0 
		(imports-from (django.db models)
			      (django.contrib.auth.models User)))
	       ))
	    
	    (python
	     (do0
	      "#export"
	      (comments "the class is a table and each element is a column in the table")
	      ,(let ((info `(setf author (models.ForeignKey User
							    :on_delete models.CASCADE)
				  created models.DateTimeField :auto_now_added True
				  modified models.DateTimeField :auto_now_added True)))
	       `(do0 (class Post (models.Model)
			(setf image (models.ImageField)
			      description (models.TextField))
			,info
			)
		     (class Comment (models.Model)
			    (setf post (models.ForeignKey Post :on_delete models.CASCADE)
				  text (models.TextField))
			    ,info)))))
	    ))
     (gen `(posts views)
	  `(
	    (python (do0
		     "#export"
	       (do0 
		(imports-from (django.views.generic.list ListView)
			      (posts.models Post)
			      )
		(class PostList (ListView)
		       (setf model Post)
		       ))
	       ))
	    ))
     (gen `(pygram urls)
	  `(
	    (python (do0
		     "#export"
	       (do0 
		(imports-from (django.contrib admin)
			      (django.urls path)
			      (django.conf settings)
			      (django.conf.urls.static static)
			      (posts.views PostList)
			      )
		(setf urlpatterns
		      (+ (list
			(path (string "admin/")
			      admin.site.urls)
			(path (string "")
			      (PostList.as_view)
			      :name (string "list")))
			 (static
			  settings.MEDIA_URL
			  :document_root settings.MEDIA_ROOT))))
	       ))
	    ))
     (gen `(posts admin)
	  `(
	    (python (do0
		     "#export"
	       (do0 
		(imports-from (django.contrib admin)
			      (posts.models Post)
			      )
		(admin.site.register Post admin.ModelAdmin))
	       ))
	    ))))

  (sb-ext:run-program "/usr/bin/sh" `("/home/martin/stage/cl-py-generator/example/78_django/source/setup01_nbdev.sh")
			)
  
  )



 

