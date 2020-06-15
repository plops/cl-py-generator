(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *repo-sub-path* "21_vulkan_qt")
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *repo-sub-path*))
  (defparameter *code-file* "run_00_show")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *inspection-facts*
    `((10 "")))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))

  
  (let* ((code
	  `(do0
	    
	    (do0
	     #+nil (do0
		    (imports (matplotlib))
                                        ;(matplotlib.use (string "QT5Agg"))
		    "from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)"
		    "from matplotlib.figure import Figure"
		    (imports ((plt matplotlib.pyplot)
			      matplotlib.colors
			      (animation matplotlib.animation)
			      (xrp xarray.plot)))
		    
		    (plt.ion)
					;(plt.ioff)
		    (setf font (dict ((string size) (string 8))))
		    (matplotlib.rc (string "font") **font)
		    )


	     
	     )
	    

	    
	    (imports (			;os
					;sys
					;traceback
					;pdb
					;time
					;docopt
					;pathlib
		      ;(yf yfinance)
		      (np numpy)
		      ;collections
					;serial
		      (pd pandas)
					;(xr xarray)
					;(xrp xarray.plot)
					;skimage.restoration 
					;skimage.feature
					;skimage.morphology
					;skimage.measure
					; (u astropy.units)
					; EP_SerialIO
					;scipy.ndimage
					;scipy.optimize
					;scipy.ndimage.morphology
					; nfft
					; ttv_driver
		      pathlib
					;re
					;requests
					;zipfile
					;io
					;sklearn
					;sklearn.linear_model
		      
		      ))
	    "from vulkan import *"
	    (do0
	     (imports (PyQt5)
		      )
	     "from PyQt5 import QtCore, QtGui, QtWidgets"
	     "from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel"
	     "from PyQt5.QtCore import QAbstractTableModel, Qt")
	    
	    (do0
	     (comment "%%")
	     (setf
	      _code_git_version
	      (string ,(let ((str 
			      #-sbcl "xxx"
			      #+sbcl (with-output-to-string (s)
				       (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
			 (subseq str 0 (1- (length str)))))
	      _code_repository (string ,(format nil "https://github.com/plops/cl-py-generator/tree/master/example/~a/source/~a.py" *repo-sub-path* *code-file*)
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
				 (- tz))))))
	    #+nil
	    (class HelloTriangleApplication (QtGui.QWindow)
		   (def __init__ (self)
		     (dot (super HelloTriangleApplication self)
			  (__init__))
		     (self.setWidth 512)
		     (self.setHeight 512)

		     "global _code_git_version, _code_generation_time"
		     (self.setTitle (dot (string "vulkan -- {} {}")
					 (format _code_generation_time
						 _code_git_version))))
		   (def __del__ (self)
		     (print (string "close"))
		     pass))

	    (do0
	     (setf validation_layer (list (string "VK_LAYER_LUNARG_standard_validation"))
		   enable_validation_layers True)
	     (class InstanceProcAddr (object)
		    (def __init__ (self func)
		      (setf self.__func func))
		    (def __call__ (self *args **kwargs)
		      (setf func_name self.__func.__name__
			    func (vkGetInstanceProcAddr (aref args 0)
							func_name))
		      (if func
			  (return (func *args **kwargs))
			  (return VK_ERROR_EXTENSION_NOT_PRESENT)))))

	    (do0
	     @InstanceProcAddr
	     (def vkCreateDebugReportCallbackEXT (instance pCreateInfo pAllocator)
	       pass)
	     @InstanceProcAddr
	     (def vkDestroyDebugReportCallbackEXT (instance pCreateInfo pAllocator)
	       pass)
	     (def debug_callback (*args)
	       (print (dot (string "debug: {} {}")
			   (format (aref args 5)
				   (aref args 6))))
	       (return 0))
	     )
	    
	    (do0
	     (setf app (QApplication (list (string "")))
		   win (QWidget)
		   ;(HelloTriangleApplication)
		   )
	     (setf appinfo (VkApplicationInfo :pApplicationName (string "python vk")
					      :applicationVersion (VK_MAKE_VERSION 1 0 0)
					      :pEngineName (string "pyvulkan")
					      :engineVersion (VK_MAKE_VERSION 1 0 0)
					      :apiVersion VK_API_VERSION)
		   extensions (list (for-generator (e (vkEnumerateInstanceExtensionProperties None))
						   e.extensionName))
		   instanceinfo (VkInstanceCreateInfo :pApplicationInfo appinfo
						      :enabledLayerCount 0
						      :enabledExtensionCount (len extensions)
						      :ppEnabledExtensionNames extensions)
		   instance (vkCreateInstance instanceinfo None))


	     (when enable_validation_layers
	       (setf createinfo (VkDebugReportCallbackCreateInfoEXT
				 :flags (logior VK_DEBUG_REPORT_WARNING_BIT_EXT
						VK_DEBUG_REPORT_ERROR_BIT_EXT)
				 :pfnCallback debug_callback))
	       (setf callback (vkCreateDebugReportCallbackEXT instance createinfo None)))
	     
	     (win.show)

	     (def cleanup ()
	       "global win, instance"
	       (vkDestroyInstance instance None)
	       (del win))

	     (app.aboutToQuit.connect cleanup)
	     
	     (def run ()
	       (sys.exit
		(app.exec_)))
	     )
	    )))
    (write-source (format nil "~a/source/~a" *path* *code-file*) code)))





