(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-change-case"))

(in-package :cl-py-generator)

(setf *features* (union *features* '(:more)))
(setf *features* (set-difference *features* '(:more)))

(let ()
  (defparameter *project* "156_scan")
  (defparameter *idx* "01") 
  (defparameter *path* (format nil "/home/martin/stage/cl-py-generator/example/~a" *project*))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key msg vars)
    `(do0				;when args.verbose
      (print (dot (string ,(format nil "~a ~{~a={}~^ ~}"
				   msg
				   (mapcar (lambda (x)
                                             (emit-py :code x))
					   vars)))
                  (format 
                          ,@vars)))))
  (defun doc (def)
    `(do0
      ,@(loop for e in def
              collect
              (destructuring-bind (&key name val (unit "-") (help name)) e
                `(do0
                  (comments ,(format nil "~a (~a)" help unit))
                  (setf ,name ,val))))))

  
  (let* ((notebook-name "scan"))
    (write-source
     (format nil "~a/source01/p~a_~a" *path* "00" notebook-name)
     `(do0
       "#!/usr/bin/env python3"
       (comments "load a slow-mo video from iphone that with structured illumination")
       (imports (
		 ;time
		 ; tqdm
					;subprocess
		 ;pathlib
		 (cv cv2)
					;argparse
		 ))
       ;(imports-from (cv2 *))

       #+nil
       (do0
	(setf parser (argparse.ArgumentParser))
	(parser.add_argument (string "input_paths")
			     :nargs (string "+")
			     :help (string "Path(s) to search for matching files."))
	,@(loop for e in `((:name "min-size"
			    :type int
			    :default 0
			    :help "Minimum size in bytes for a file to be selected.")
			   )
		collect
		(destructuring-bind (&key name default type help) e
		  (let ((cmd `(parser.add_argument (string ,(format nil "--~a" name)))))
		    (when type
		      (setf cmd (append cmd `(:type ,type))))
		    (when default
		      (setf cmd (append cmd `(:default ,default))))
		    (when help
		      (setf cmd (append cmd `(:help (string ,help)))))
		    cmd)))
	(setf args (parser.parse_args)))

       (comments "load video file")
       (setf fn (string "IMG_3355.mov"))
       (setf v (cv.VideoCapture fn))
       (unless (v.isOpened)
	 ,(lprint :msg "video could not be opened" :vars `(fn)))
       (setf w (v.get cv.CAP_PROP_FRAME_WIDTH)
	     h (v.get cv.CAP_PROP_FRAME_HEIGHT))
       (setf win (string "screen"))
       (cv.namedWindow win
		       cv.WINDOW_AUTOSIZE)
       (while True
	      (setf (ntuple a frame)
		    (v.read))
	      (when (is frame None)
		break)
	      (cv.imshow win frame)
	      (setf k (cv.waitKey 30))
	      (when (==   k 27)
		break)
	      )
       
       )))
  )
