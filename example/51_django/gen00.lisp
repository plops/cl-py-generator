(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/51_django/")
  
  (write-source
   (format nil "~a/mysite/polls/views" *path*)
   `(do0
     (imports-from (django.http HttpResponse HttpResponseRedirect Http404)
					;(django.template loader)
		   (django.shortcuts render get_object_or_404)
		   (django.urls reverse)
		   (django.views generic)
		   (django.utils timezone)
		   (.models Question Choice))
     

     ,@(loop for (stem class parent code) in `((index IndexView ListView
						      (do0 (setf context_object_name
								 (string "latest_question_list"))
							   (def get_queryset (self)
							     (return
							       (aref (dot Question
									  objects
									  (filter :pub_date__lte (timezone.now))
									  (order_by
									   (string "-pub_date")))
								     (slice "" 5))))))
					       (detail DetailView DetailView (do0 (setf model Question)
										  (def get_queryset (self)
										    (return
										      (dot Question
											   objects
											   (filter :pub_date__lte (timezone.now))
											   )))))
					       (results ResultsView DetailView (setf model Question)))
	     collect
	     `(class ,class ((dot generic ,parent))
		     (setf template_name (string ,(format nil "polls/~a.html" stem)))
		     ,code))
     
     (def vote (request question_id)
       (do0
	(setf question (get_object_or_404 Question :pk question_id))
	)
       (try
	(setf selected_choice (question.choice_set.get :pk (dot request (aref POST (string "choice")))))
	((tuple KeyError Choice.DoesNotExist)
	 (comments "redisplay voting form")
	 (return (render request (string "polls/detail.html")
			 (dictionary :question question
				     :error_message (string "you didn't select a choice")
				     ))))
	(else (setf selected_choice.votes (+ 1 selected_choice.votes))
	   (selected_choice.save)
	   ;; redirect prevents data being posted twice if user hits back button
	   ;; reverse avoids hardcoded url, name of the view we want to pass control to
	   (return (HttpResponseRedirect (reverse (string "polls:results")
						  :args (tuple question.id))))
	   ))
       )))
  (write-source
   (format nil "~a/mysite/polls/urls" *path*)
   `(do0
     (import-from django.urls path)
     (import-from "." views)
     (setf app_name (string "polls"))
     (setf urlpatterns
	   (list (path (string "")
		       (dot views IndexView (as_view))
		       :name (string "index"))
		 ,@(loop for (e f) in `((detail DetailView) (results ResultsView)
					)
			 collect
			 `(path (string ,(format nil "<int:pk>/~a" (case e
								     ('detail "")
								     (t e))))
				,(if (eq e 'vote)
				     `(dot views ,f)
				     `(dot views ,f (as_view)))
				:name (string ,e)))
		 (path (string "<int:question_id>/vote/")
		       views.vote
		       :name (string "vote"))))))
  (write-source
   (format nil "~a/mysite/polls/models" *path*)
   `(do0
     (imports (datetime))
     (imports-from (django.db models)
		  (django.utils timezone))
     (class Question (models.Model)
	    (setf question_text (models.CharField :max_length 200)
		  ;; human readable name as first arg:
		  pub_date (models.DateTimeField (string "date published")))
	    (def __str__ (self)
	      (return self.question_text))
	    (def was_published_recently (self)
	      (setf now (timezone.now))
	      (return (<= (- now
			     (datetime.timedelta :days 1))
			  self.pub_date
			  now))))
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
     (imports-from (django.contrib admin)
		   (.models Question Choice))

     (class ChoiceInline (admin.TabularInline)
	    (setf model Choice
		  extra 3))
     
     (class QuestionAdmin (admin.ModelAdmin)
	    #+nil (do0
	     (comments "publication date field before question field")
	     (setf fields (list (string "pub_date")
				(string "question_text"))))
	    (do0
	     (setf fieldsets (list
			      (tuple None (dictionary :fields (list (string "question_text"))))
			      (tuple (string "date information") (dictionary :fields (list (string "pub_date"))))))
	     (setf inlines (list ChoiceInline))
	     (setf list_display (tuple (string "question_text")
				       (string "pub_date")
				       (string "was_published_recently")))))
     (admin.site.register Question QuestionAdmin)))
   (write-source
   (format nil "~a/mysite/polls/tests" *path*)
   `(do0
     (imports (datetime))
     (imports-from (django.test TestCase)
		   (django.utils timezone)
		   (django.urls reverse)
		   (.models Question))
     (def create_question (question_text days)
       (string3 "create question. days is negative for publishing date in the past")
       (setf time (+ (timezone.now)
		     (datetime.timedelta :days days)))
       (return (Question.objects.create
		:question_text question_text
		:pub_date time)))
     (class QuestionDetailViewTests (TestCase)
	    ,@(loop for e in `((past :pre  (setf q (create_question :question_text (string "past question")
								      :days -5)
						   )
				       :post (self.assertContains response q.question_text) )
			       (future :pre  (setf q (create_question :question_text (string "future question")
								      :days 5))
				       :post (self.assertEqual response.status_code 404) ))
		    collect
		    (destructuring-bind (name &key pre post) e
		      `(def ,(format nil "test_~a_question" name) (self)
			 ,(if pre
			      pre
			      `(comments "no pre"))
			 (setf url (reverse (string "polls:detail")
					    :args (tuple q.id))
			       response (self.client.get url))
			 ,(if post
			      post
			      `(comments "no post"))
			 ))))
     (class QuestionIndexViewTests (TestCase)
	    ,@(loop for e in `((no :post (do0
					  (self.assertEqual response.status_code 200)
					  (self.assertContains response (string "no polls are available"))
					  (self.assertQuerysetEqual (aref response.context
									  (string "latest_question_list"))
								    (list))))
			       (past :pre (setf q (create_question :question_text (string "past question")
								   :days -30))
				     :post (self.assertQuerysetEqual (aref response.context (string "latest_question_list"))
								     (list q)))
			       (future :pre (setf q (create_question :question_text (string "futurue question")
								   :days 30))
				       :post (do0
					      (self.assertContains response (string "no polls are available"))
					      (self.assertQuerysetEqual (aref response.context (string "latest_question_list"))
								      (list ))))
			       (future_and_past :pre (setf q0 (create_question :question_text (string "past question")
									       :days -30)
							    q1 (create_question :question_text (string "futurue question")
								   :days 30))
				     :post (self.assertQuerysetEqual (aref response.context (string "latest_question_list"))
								     (list q0)))
			       (two_past :pre (setf q0 (create_question :question_text (string "past question")
									       :days -30)
							    q1 (create_question :question_text (string "futurue question")
								   :days -5))
				     :post (self.assertQuerysetEqual (aref response.context (string "latest_question_list"))
								     (list q1 q0)))
			       )
		    collect
		    (destructuring-bind (name &key pre post) e
		      `(def ,(format nil "test_~a_question" name) (self)
			 ,(if pre
			      pre
			      `(comments "no pre"))
			 (setf response (self.client.get (reverse (string "polls:index"))))
			 ,(if post
			      post
			      `(comments "no post"))))))
     (class QuestionModelTests (TestCase)
	    ,@(loop for (name code result) in
		    `((future (+ (timezone.now)
				 (datetime.timedelta :days 30))
			      False)
		      (old (- (timezone.now)
				 (datetime.timedelta :days 1 :seconds 1))
			   False)
		      (recent (- (timezone.now)
				 (datetime.timedelta :hours 23 :minutes 59 :seconds 59))
			      True))
		    collect
		    `(def ,(format nil "test_was_published_recently_with_~a_question" name) (self)
	      (setf time ,code
		    question (Question :pub_date time))
	      (self.assertIs (question.was_published_recently)
			     ,result))
		    )
	    )
     ))
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



