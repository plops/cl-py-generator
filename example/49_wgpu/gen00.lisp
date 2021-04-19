(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))
(in-package :cl-py-generator)



(progn
  (defparameter *path* "/home/martin/stage/cl-py-generator/example/49_wgpu")
  (defparameter *code-file* "run_00_start")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *host*
    "10.1.99.12")
  (defparameter *inspection-facts*
    `((10 "")))

  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  (let* ((code
	  `(do0
	    (do0 
		 #-nil(do0
		  
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
	         (imports (		;os
					;sys
					time
					;docopt
					pathlib
					;(np numpy)
					;serial
					(pd pandas)
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
			  ; (np numpy)
			   ;(cv cv2)
					;(mp mediapipe)
					;jax
					; jax.random
					;jax.config
			  ; copy

			   
			   ;cProfile
			   ;(np jax.numpy)
			   ;jax
			   ;jax.random
					;jax.config
			   glfw
			   wgpu
			   
			   
			   wgpu.gui.glfw
			   wgpu.backends.rs
			   ))
		 ,(format nil "from pyshader import ~{~a~^, ~}" `(python2shader RES_INPUT RES_OUTPUT vec2 vec3 vec4 i32))
		 ;"from jax import grad, jit, jacfwd, jacrev, vmap, lax, random"
		 ;,(format nil "from jax.numpy import ~{~a~^, ~}" `(sqrt newaxis sinc abs))
		 ;,(format nil "from jax import ~{~a~^, ~}" `(grad jit jacfwd jacrev vmap lax random))
		 #+nil (jax.config.update (string "jax_enable_x64")
				    True)
		 
		 (do0
		  (comments "https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions")
		  (class Timer (object)
			 (def __init__ (self &key (name None))
			   (setf self.name name))
			 (def __enter__ (self)
			   (setf self.tstart (time.time)))
			 (def __exit__ (self type value traceback)
			   (print (dot (string "[{}] elapsed: {}s")
				       (format self.name (- (time.time)
							    self.tstart)))))))

		 ;; m = buffer.read_data()
		 ;; m.cast('f')
		 ;; np.frombuffer(m, np.float32)
		 ;; https://github.com/pygfx/wgpu-py/blob/main/examples/triangle.py
		 (do0
		  (def main (canvas)
		    (setf adapter (wgpu.request_adapter :canvas canvas
							:power_preference (string "high-performance"))
			  device (adapter.request_device))
		    (return (_main canvas device)))
		  (do0 "@python2shader"
		       (def vertex_shader (&key (index (tuple RES_INPUT (string "VertexId") i32))
						(pos (tuple RES_OUTPUT (string "Position") vec4))
						(color (tuple RES_OUTPUT 0 vec3)))
			 (setf positions (list (vec2 .0 .5)
					       (vec2 .5 .5)
					       (vec2 -.5 .7))
			       p (aref positions index)
			       pos (vec4 p .0 1.)
			       color (vec3 p .5))))
		  (do0 "@python2shader"
		       (def fragment_shader (&key (in_color (tuple RES_INPUT 0 vec3))
						  (out_color (tuple RES_OUTPUT 0 vec4)))
			 (setf out_color (vec4 in_color 1.0))))
		  (def _main (canvas device)
		    ,@(loop for (obj code) in `((vshader vertex_shader)
					(fshader fragment_shader))
			    collect
			    `(setf ,obj (device.create_shader_module :code ,code)))
		    ,@(loop for e in `((bind_group_layout :entries (list))
				       (bind_group :layout bind_group_layout :entries (list))
				       (pipeline_layout :bind_group_layout (list bind_group_layout))
				       (render_pipeline :layout pipeline_layout))
			    collect
			    (destructuring-bind (name &rest rest) e
			      `(setf ,name (dot device (,(format nil "create_~a" name)
							,@rest))))))
		  )
		 
		 (do0
		  (glfw.init)
		  (setf glfw.ERROR_REPORTING (string "warn"))
		  (setf canvas (wgpu.gui.glfw.WgpuCanvas :title (string "wgpu triangle with glfw")))
		  (main canvas)
		  (while (wgpu.gui.glfw.update_glfw_canvasses)
			 (glfw.poll_events))
		  (glfw.terminate))
		 
		 )
	    ))
	 
	 )
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))



