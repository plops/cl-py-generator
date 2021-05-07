(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/51_django/")
  
  (write-source
   (format nil "~a/mysite/polls/views" *path*)
   `(do0
     (imports-from (django.http HttpResponse Http404)
		   ;(django.template loader)
      (django.shortcuts render get_object_or_404)
      (.models Question))
  
     (def index (request)
       (setf latest_question_list (aref (Question.objects.order_by
					 (string "-pub_date"))
					(slice "" 5))
	     ;template (loader.get_template (string "polls/index.html"))
	     context (dictionary
		      :latest_question_list latest_question_list))
       (return ;(HttpResponse (template.render context request))
	 (render request (string "polls/index.html") context)))
     (def detail (request question_id)
       (do0
	(setf question (get_object_or_404 Question :pk question_id))
	)
       #+nil (try
	(setf question (Question.objects.get :pk question_id))
	(Question.DoesNotExist
	 (raise Http404 (string "question does not exist"))))
       (return (render request (string "polls/detail.html")
		       (dictionary :question question))))
     ,@(loop for e in `( results vote)
	     collect
	     `(def ,e (request question_id)
		(return (HttpResponse (dot (string ,(format nil "~a of question {}." e))
					   (format question_id))))))))
  (write-source
   (format nil "~a/mysite/polls/urls" *path*)
   `(do0
     (import-from django.urls path)
     (import-from "." views)
     
     (setf urlpatterns
	   (list (path (string "")
		       views.index
		       :name (string "index"))
		 ,@(loop for e in `(detail results vote)
			 collect
			 `(path (string ,(format nil "<int:question_id>/~a" (case e
									      ('detail "")
									      (t e))))
				(dot views ,e)
				:name (string ,e)))))))
  (write-source
   (format nil "~a/mysite/polls/models" *path*)
   `(do0
     (import-from django.db models)
     (class Question (models.Model)
	    (setf question_text (models.CharField :max_length 200)
		  ;; human readable name as first arg:
		  pub_date (models.DateTimeField (string "date published")))
	    (def __str__ (self)
	      (return self.question_text))
	    (def was_published_recently (self)
	      (return (<= (- (timezone.now)
			     (datetime.timedelta :days 1))
			  self.pub_date))))
     (class Choice (models.Model)
	    (setf question (models.ForeignKey Question
					      :on_delete models.CASCADE)
		  choice_text (models.CharField :max_length 200)
		  votes (models.IntegerField :default 0))
	    (def __str__ (self)
	      (return self.choice_text)))))
  (write-source
   (format nil "~a/mysite/polls/admin" *path*)
   `(do0
     (import-from django.contrib admin)
     (import-from .models Question)
     (admin.site.register Question)))
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
	    (string "polls.apps.PollsConfig")
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
				 (string ,(format nil "django.contrib.auth.password_validation.~a" e))))))
     (setf LANGUAGE_CODE (string "en-us")
	   TIME_ZONE (string "Europe/Amsterdam")
	   USE_I18N (string "True")
	   USE_L10N (string "True")
	   USE_TZ (string "True"))
     (setf STATIC_URL (string "/static/"))
     (setf DEFAULT_AUTO_FIELD (string "django.db.models.BigAutoField"))
     )))



