(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/40_kivy")
  (defparameter *code-file* "run_00_kivy")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

     
  (let* (
	 
	 (code
	  `(do0
	    (do0 
		 #+nil(do0
		  
		  (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
					;"from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
					;"from matplotlib.figure import Figure"
		  (imports ((plt matplotlib.pyplot)
			  ;  (animation matplotlib.animation) 
                            ;(xrp xarray.plot)
			    ))
                  
		  (plt.ion)
					;(plt.ioff)
		  ;;(setf font (dict ((string size) (string 6))))
		  ;; (matplotlib.rc (string "font") **font)
		  )
		#+nil  (imports (		;os
					;sys
					;time
					;docopt
					;pathlib
					;(np numpy)
					;serial
					;(pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration
					;(u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;   scipy.optimize
					;nfft
					;sklearn
					;sklearn.linear_model
					;itertools
					;datetime
			   (np numpy)
			   (cv cv2)
					;(mp mediapipe)
					;jax
			   ; jax.random
			   ;jax.config
			   copy
			   
			   ))
		,@(loop for (pkg exprs) in `((kivy.app (App))
					     (kivy.uix.widget (Widget))
					     (kivy.properties (NumericProperty
							       ReferenceListProperty
							       ObjectProperty)
							      )
					     (kivy.vector (Vector)))
			collect
			(format nil "from ~a import ~{~a~^, ~}" pkg exprs))
		 (setf
		  _code_git_version
		  (string ,(let ((str (with-output-to-string (s)
					(sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			     (subseq str 0 (1- (length str)))))
		  _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py")
					   )

		  _code_generation_time
		  (string ,(multiple-value-bind
				 (second minute hour date month year day-of-week dst-p tz)
			       (get-decoded-time)
			     (declare (ignorable dst-p))
			     (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
				     hour
				     minute
				     second
				     (nth day-of-week *day-names*)
				     year
				     month
				     date
				     (- tz))
			     )))
		 (class PongBall (Widget)
			(setf vx (NumericProperty 0)
			      vy (NumericProperty 0)
			      v (ReferenceListProperty vx vy))
			(def move (self)
			  (setf self.pos
				(+ (Vector *self.v)
				   self.pos))))
		 (class PongGame (Widget)
			(setf ball (ObjectProperty None))
			(def serve_ball (self)
			  (setf self.ball.center self.center
				self.ball.v (dot
					     (Vector 4 0)
					     (rotate (randint 0 360)))))
			(def update (self dt)
			  (do0
			   (self.ball.move)
			   (when (or (< self.ball.y 0)
				     (< self.height self.ball.top))
			     (setf self.ball.vy (* -1 self.ball.vy)))
			   (when (or (< self.ball.x 0)
				     (< self.width self.ball.right))
			     (setf self.ball.vx (* -1 self.ball.vx))))
			  pass))
		 (class PongApp (App)
			(def build (self)
			  (setf game (PongGame))
			  (Clock.schedule_interval game.update
						   (/ 1s0 60))
			  (return game)))
		 
		 (when (== __name__ (string "__main__"))
		   (dot (PongApp)
			(run)))))))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)
    (with-output-to-file (s (format nil "~a/source/pong.kv" *path*)
			    :if-exists :supersede
			    :if-does-not-exist :create)
      (format s "#:kivy 1.0.9


<PongBall>:
    size: 50, 50 
    canvas:
        Ellipse:
            pos: self.pos
            size: self.size          

<PongGame>:
    ball: pong_ball
    canvas:
        Rectangle:
            pos: self.center_x - 5, 0
            size: 10, self.height
    
    Label:
        font_size: 70  
        center_x: root.width / 4
        top: root.top - 50
        text: "0"
        
    Label:
        font_size: 70  
        center_x: root.width * 3 / 4
        top: root.top - 50
        text: "0"
    
    PongBall:
        id: pong_ball
        center: self.parent.center
        
  "))))

