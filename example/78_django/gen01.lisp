;; NOTE: loading lass with :invert readtable gives an error, use ACCEPT, this works for me as of 2022-01-29
(eval-when (:compile-toplevel :execute :load-toplevel)
  (setf (readtable-case *readtable*) :upcase)
  (ql:quickload "lass")
  (ql:quickload "serapeum")
  (ql:quickload "spinneret"))

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn 
  (setf *warn-breaking* nil)
  ;; the following code needs inverted readtable, otherwise symbols
  ;; and filenames may have the wrong case and everything breaks in
  ;; horrible ways
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/78_django")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (let ((shell-counter 1))
    (flet ((out (fn lines &key (count t))
	     
	     (with-open-file (s (if count
				    (format nil "~a/source/web/setup~3,'0d_~a" *path* shell-counter fn)
				    (format nil "~a/source/web/~a" *path* fn))
				:direction :output
				:if-exists :supersede
				:if-does-not-exist :create)
	       (format s "~{~a~%~}" lines))
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
	     "# undo: rm -rf env"))
      (out "startproject.sh"
	   `("# bootstrap django project"
	     "django-admin startproject pygram ."
	     "# undo: rm -rf pygram manage.py"))
      (out "runserver.sh"
	   `("# start django webserver"
	     "./manage.py runserver"))
      (out "startapp_posts.sh"
	   `("# add django application with business logic for creating andd commentiiing on posts"
	     "./manage.py startapp posts"
	     "# undo: rm -rf posts"))
      (out "makemigrations.sh"
	   `("# tell django to make migrations"
	     "./manage.py makemigrations"
	     "# undo: rm -rf posts/migrations "))
      (out "migrate.sh"
	   `("# tell django to make migrations"
	     "./manage.py migrate"
	     "# undo: rm -rf posts/migrations "))
      (out "sqlite_tables.sh"
	   `("# look at the tables in the database"
	     "sqlite3 db.sqlite3 .tables"
	     "# undo: n.a."))
      (out "manage_createsuperuser.sh"
	   `("# look at the tables in the database"
	     "./manage.py createsuperuser"
	     "echo 'now try to login with your new user at http://127.0.0.1:8000/admin'"
	     "# undo: ")))) 
  (defmacro with-page ((&key title) &body body)
    "create html page with inline css"
    `(spinneret:with-html-string
       (:doctype)
       (:html
	(:head
	 (:title ,title)
	 (:style :type "text/css"
		 (:raw
		  (lass:compile-and-write
		   `(body :font-family "sans-serif")
		   `(.container :width 25% :margin auto)
		   `(.header :patting 15px
			     :text-align center
			     :font-size 2em
			     :background "#f2f2f2"
			     :margin-bottom 15px)
		   `(.header>a   :color inherit
				 :text-decoration none)
		   ))))
	(:body ,@body))))
  (defmacro with-template ((&key path base)  &body body)
    "create html page by extending a base template"
    `(gen-html ,path
	       (spinneret:with-html-string
		 (:raw ,(format nil "{% extends '~a' %}" base))
		 (:raw "{% block content %}")
		 ,@body
		 (:raw "{% endblock content %}"))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create python file in a directory below source/web"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}" path))
					    ))) code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter))
	   (gen-html (path code)
	     "create html file in a directory below source/webqq"
	     (let ((fn (format nil "~a/source/web/~{~a~^/~}" *path* path)))
	       (ensure-directories-exist (pathname fn))
	       (with-open-file (s fn
				  :direction :output
				  :if-exists :supersede
				  :if-does-not-exist :create)
		 (format s "~a" code))
	       (format t "~&~c[31m wrote HTML ~c[0m ~a~%" #\ESC #\ESC fn))))
      (gen-html `(posts templates posts base.html)
		(with-page (:title "PyGram")
		  (:div :class "header"
			(:a :href "/"
			    "PyGram"))
		  (:div :class "container"
			(:raw "{% block content %}{% endblock %}"))))
      (with-template (:path `(posts templates posts post_list.html)
			    :base posts/base.html)
	(:a :href "{% url 'new' %}"
	    (:h1 "Create new post"))
	(:raw "{% for post in object_list %}")
	(:strong "{{ post.author.username }}")
	(:br)
	(:img :src "{{ post.image.url }}"
	      :width 400
	      :height 400)
	(:p (:em "{{ post.created }}")
	    (:br)
	    "{{ post.description }}")
	(:raw "{% endfor %}"))
      
      (with-template (:path `(posts templates posts post_form.html)
			    :base posts/base.html)
	(:h1 "Create new post")
	(:form :action "{% url 'new' %}"
	       :method "POST"
	       :enctype "multipart/form-data"
	       "{% csrf_token %}"
	       "{{ form.as_p }}"
	       (:input :type "submit" :value "Post")))

      (with-template (:path `(posts templates posts post_detail.html)
			    :base posts/base.html)
	(:strong "{{ object.author.username }}")
	(:br)
	(:img :src "{{ object.image.url }}"
	      :width 200
	      :height 200)
	(:p (:em "{{ object.created }}")
	    (:br))
	(:ul
	 "{% for comment in object.comment_set.all %}"
	 (:li
	  (:strong "{{ comment.author }}")
	  (:br)
	  "{{ comment.text }}")
	 "{% endfor %}")
	(:form :action "{% url 'detail' pk=object.id %}"
	       :method "POST"
	       "{% csrf_token %}" ;; cross site request forgery protection
	       "{{ comment_form.as_p }}"
	       (:input :type "submit"
		       :value "Comment")))
      (gen `(posts models)
	   `((python
	      (do0
	       (comments "export")
	       (imports-from (django.db models)
			     (django.contrib.auth.models User))))
	     (python
	      (cell
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
			       ,info)))))))
      (gen `(posts views)
	   `((python (cell
		      (do0 
		       (imports-from ,@(loop for (e f) in `((list ListView)
							    (edit CreateView)
							    (detail DetailView))
					     collect
					     `((dot django views generic ,e) ,f))
				     (django forms)
				     (django.shortcuts redirect)
				     (django.urls reverse)
				     (posts.models Post Comment)
				     )
		       (class PostList (ListView)
			      (setf model Post)
			      )
		       (class PostCreate (CreateView)
			      (setf model Post
				    fields (list (string "image")
						 (string "description")
						 (string "author"))
				    success_url (string "/")))
		       (class CommentForm (forms.Form) 
			      (setf comment (forms.CharField)))
		       (class PostDetail (DetailView)
			      (setf model Post)
			      (def get_context_data (self *args **kwargs)
				   (setf context (dot (super)
						      (get_context_data *args **kwargs))
					 (aref context (string "comment_form")) (CommentForm))
				   (return context))
			      (def post (self request *args **kwargs)
				   (setf comment_form (CommentForm request.POST))
				   (if (comment_form.is_valid)
				       ;; author is the user who made the request
				       (do0 (setf comment (Comment :author request.user
								   :post (self.get_object)
								   :text (dot comment_form
									      (aref cleaned_data (string "comment")))))
					    (comment.save))
				       (do0
					;; bad error handling for now
					(raise Exception)))
				   (return (redirect (reverse (string "detail")
							      :args (list (dot self
									       (get_object)
									       id))))))))))))
      (gen `(pygram urls)
	   `((python (cell
		      (do0 
		       (imports-from (django.contrib admin)
				     (django.urls path)
				     (django.conf settings)
				     (django.conf.urls.static static)
				     (posts.views PostList PostCreate PostDetail)
				     )
		       (setf urlpatterns
			     (+ (list
				 (path (string "admin/")
				       admin.site.urls)
				 ,@(loop for e in `((:url "" :class PostList :name list)
						    (:url "new/" :class PostCreate :name new)
						    (:url "posts/<pk>/" :class PostDetail :name detail))
					 collect
					 (destructuring-bind (&key url class name) e
					   `(path (string ,url)
						  (dot ,class (as_view))
						  :name (string ,name)))))
				(static settings.MEDIA_URL
					:document_root settings.MEDIA_ROOT))))))))
      (gen `(posts admin)
	   `((python (cell
		      (do0 
		       (imports-from (django.contrib admin)
				     (posts.models Post))
		       (admin.site.register Post admin.ModelAdmin))))))
      (gen `(tests functional)
	   `((python (cell
		      (do0
		       (comments "pip3 install --user selenium"
				 "https://github.com/mozilla/geckodriver/releases"
				 "wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz"
				 "tar xaf geckodriver*.tar.gz"
				 "sudo cp geckodriver /usr/local/bin"
				 "follow textbook 2017 Test-Driven Development with Python ")
		       (imports (unittest))
		       (imports-from (selenium webdriver))

		       (class NewVisitorTest (unittest.TestCase)
			      (def  setUp (self)
				(setf self.browser (webdriver.Firefox)))
			      (def tearDown (self)
				(self.browser.quit))
			      ;; any method starting with test will be
			      ;; run by the test runner
			      (def test_can_create_post_and_retrieve_it_later (self)
				(self.browser.get (string "http://localhost:8000"))
				(self.assertIn (string "PyGram") self.browser.title)
				(self.fail (string "Finish the test!"))))
		       (when (== __name__ (string "__main__"))
			 (unittest.main :warnings (string "ignore"))))))))))
  (sb-ext:run-program "/usr/bin/sh"
		      `("/home/martin/stage/cl-py-generator/example/78_django/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))



 
