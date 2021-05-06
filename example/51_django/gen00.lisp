(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/51_django/")

  (write-source
   (format nil "~a/mysite/polls/views" *path*)
   `(do0
     (imports (django.http))
     (def index (request)
       (return (django.http.HttpResponse (string "hello world"))))))
  (write-source
   (format nil "~a/mysite/polls/urls" *path*)
   `(do0
     (import-from django.urls path)
     (import-from "." views)
     
     (setf urlpatterns
	   (list (path (string "")
		       views.index
		       :name (string "index"))))))
  (write-source
   (format nil "~a/mysite/mysite/urls" *path*)
   `(do0
     (import-from django.urls path include)
     (import-from django.contrib admin)

     
     
     (setf urlpatterns
	   (list (path (string "polls/")
		       (include (string "polls.urls")))
		 (path (string "admin/")
		       admin.site.urls)))))

  (write-source
   (format nil "~a/mysite/mysite/settings" *path*)
   `(do0
     (import-from pathlib Path)
     (setf BASE_DIR (dot (Path __file__)
			 (resolve)
			 parent
			 parent)
	   SECRET_KEY (string ,(with-open-file (s "secret")
				 (let ((a (make-string (file-length s))))
				   (read-sequence a s)
				   a)))
	   DEBUG True
	   ALLOWED_HOSTS (list))
     (setf INSTALLED_APPS
	   (list
	    ,@(loop for e in `(admin auth contenttypes sessions messages staticfiles)
		    collect
		    `(string ,(format nil "django.contrib.~a" e)))))
     (setf MIDDLEWARE
	   (list
	    ,@(loop for e in `(middleware.security.Security
			       contrib.sessions.middleware.Session
			       middleware.common.Common
			       middleware.csrf.CsrfView
			       contrib.auth.middleware.Authentication
			       contrib.messages.middleware.Message
			       middleware.clickjacking.XFrameOptions)
		    collect
		    `(string ,(format nil "django.~aMiddleware" e)))))
     (setf ROOT_URLCONF (string "mysite.urls"))
     (setf TEMPLATES (list
		      (dictionary :BACKEND (string "django.template.backends.django.DjangoTemplates")
				  :DIRS (list)
				  :APP_DIRS True
				  :OPTIONS (dictionary :context_processors
						       (list (string "django.template.context_processors.debug")
							     (string "django.template.context_processors.request")
							     (string "django.contrib.auth.context_processors.auth")
							     (string "django.contrib.messages.context_processors.messages"))))))
     (setf WSGI_APPLICATION (string "mysite.wsgi.application"))
     (setf DATABASES
	   (dictionary :default
		       (dictionary :ENGINE (string "django.db.backends.sqlite3")
				   :NAME (/ BASE_DIR (string "db.sqlite3")))))
     (setf AUTH_PASSWORD_VALIDATORS
	   (list
	    ,@(loop for e in `(UserAttributeSimilarityValidator
			       MinimumLengthValidator
			       CommonPasswordValidator
			       NumericPasswordValidator)
		    collect
		    `(dictionary :NAME
				 (string ,(format nil "django.auth.password_validation.~a" e))))))
     (setf LANGUAGE_CODE (string "en-us")
	   TIME_ZONE (string "UTC")
	   USE_I18N (string "True")
	   USE_L10N (string "True")
	   USE_TZ (string "True"))
     (setf STATIC_URL (string "/static/"))
     (setf DEFAULT_AUTO_FIELD (string "django.db.models.BigAutoField"))
     )))



