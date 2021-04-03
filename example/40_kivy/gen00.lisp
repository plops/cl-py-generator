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
		(imports (random))
		,@(loop for (pkg exprs) in `((kivy.app (App))
					     (kivy.uix.widget (Widget))
					     (kivy.properties (NumericProperty
							       ReferenceListProperty
							       ObjectProperty)
							      )
					     (kivy.vector (Vector))
					     (kivy.clock (Clock)))
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
		 (class PongPaddle (Widget)
			(setf score (NumericProperty 0))
			(def bounce_ball (self ball)
			  (when (self.collide_widget ball)
			    (setf (ntuple vx vy) ball.v
				  offset (/ (- ball.center_y
					       self.center_y)
					    (* .5 self.height))
				  bounced (Vector (* -1 vx)
						  vy)
				  vel (* 1.1 bounced)
				  ball.velocity (ntuple vel.x
							(+ vel.y offset)))
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
			,@(loop for e in `(ball player1 player2)
				collect
				`(setf ,e (ObjectProperty None)))
			(def serve_ball (self &key (vel (tuple 4 0)))
			  (setf self.ball.center self.center
				self.ball.v vel))
			(def update (self dt)
			  (do0
			   (self.ball.move)
			   ,@(loop for e in `(player1
					      player2)
				   collect
				   `((dot self
					  ,e
					  bounce_ball)
				     self.ball))
			   (when (or (< self.ball.y 0)
				     (< self.height self.ball.top))
			     (setf self.ball.vy (* -1 self.ball.vy)))
			   #+nil (when (or (< self.ball.x 0)
				     (< self.width self.ball.right))
				   (setf self.ball.vx (* -1 self.ball.vx)))
			   ,@(loop for (e f) in `((player2 (< self.ball.x
							      self.x))
						  (player1 (< self.width
							      self.x)))
				   collect
				   `(when ,f
				      (incf (dot self ,e  score))
				      (self.serve_ball :vel (tuple 4 0)))))
			  )
			(def on_touch_move (self touch)
			  (when (< touch.x (/ self.width 3))
			    (setf self.player1.center_y touch.y))
			  (when (< (- self.width (/ self.width 3)) touch.x )
			    (setf self.player2.center_y touch.y))))
		 (class PongApp (App)
			(setf game None)
			(def build (self)
			  (setf self.game (PongGame))
			  (self.game.serve_ball)
			  (Clock.schedule_interval self.game.update
						   (/ 1s0 60))
			  (return self.game)))
		 
		 (when (== __name__ (string "__main__"))
		   (setf app (PongApp))
		   (dot app
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

<PongPaddle>:
    size: 25, 200
    canvas:
        Rectangle:
            pos: self.pos
            size: self.size

<PongGame>:

    ball: pong_ball        
    player1: player_left
    player2: player_right
    canvas:
        Rectangle:
            pos: self.center_x - 5, 0
            size: 10, self.height
      
    Label:
        font_size: 70  
        center_x: root.width / 4
        top: root.top - 50
        text: str(root.player1.score)
        
    Label:
        font_size: 70  
        center_x: root.width * 3 / 4
        top: root.top - 50
        text: str(root.player2.score)
    
    PongBall:
        id: pong_ball
        center: self.parent.center
        
    PongPaddle:
        id: player_left
        x: root.x
        center_y: root.center_y
        
    PongPaddle:
        id: player_right
        x: root.width - self.width
        center_y: root.center_y

  "))))

