
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)


;; wget https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h
(progn
  (defparameter *source-dir*
					; "/home/martin/src/my_fancy_app_name/main/"
    #P"example/103_co2_sensor/source03/"
    )
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))


  (defparameter *full-source-dir*
					;"/home/martin/src/my_fancy_app_name/main/"
    #-nil (asdf:system-relative-pathname
	   'cl-py-generator
	   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (write-source
   (asdf:system-relative-pathname
    'cl-py-generator
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     "#define FMT_HEADER_ONLY"
     (include<> deque
		random
		vector
		chrono
		cmath)

    ; "#define WITHOUT_NUMPY"
     (include "core.h"
	      "matplotlibcpp.h")

     "namespace plt = matplotlibcpp;"

     ,@(let ((n-fifo 240))
	 (loop for e in `((N_FIFO ,n-fifo)
			  (RANSAC_MAX_ITERATIONS ,(max n-fifo 12))
			  (RANSAC_INLIER_THRESHOLD 0.1 :type float)
			  (RANSAC_MIN_INLIERS ,(floor (* .1 n-fifo))))
	       collect
	       (destructuring-bind (name val &key (type 'int)) e
		 (format nil "const ~a ~a = ~a;" type name val))))
     
     (defstruct0 Point2D
	 (x double)
       (y double))
     "std::deque<Point2D> fifo(N_FIFO,{0.0,0.0});"


     (defclass+ Line ()
       "public:"
       (defmethod Line (m b)
	 (declare (type double m b)
		  (construct (m_ m) (b_ b))
		  (values :constructor))
	 )
       (defmethod point (x)
	 (declare (type double x)
		  (values Point2D))
	 (return (curly x (+ (* m_ x)
			     b_))))
       "private:"
       "double m_;"
       "double b_;")
     

     (defun distance (p m b)
       (declare (type Point2D p)
		(type double m b)
		(values double))
       (do0
	(comments "division normalizes distance, so that it is independent of the slope of the line ")
	(return (/ (abs (- p.y
			   (+ (* m p.x)
			      b)))
		   (sqrt (+ 1 (* m m)))))))


     (defun ransac_line_fit (data m b inliers)
       (declare (type "std::deque<Point2D>&" data)
		(type double& m b)
		(type "std::vector<Point2D>&" inliers))
       (when (< (fifo.size) 2)
	 (return))
       "std::random_device rd;"
       (let (
	     (gen (std--mt19937 (rd)))
	     (distrib (std--uniform_int_distribution<> 0 (- (data.size)
							    1)))
	     (best_inliers (std--vector<Point2D>))
	     (best_m 0d0)
	     (best_b 0d0))
	 (dotimes (i RANSAC_MAX_ITERATIONS)
	   (comments "line model needs two points, so randomly select two points and compute model parameters")
	   (let ((idx1 (distrib gen))
		 (idx2 (distrib gen))
		 )
	     (while (== idx1 idx2)
		    (setf idx1 (distrib gen)))
	     (let ((p1 (aref data idx1))
		   (p2 (aref data idx2))
		   (m (/ (- p2.y p1.y)
			 (- p2.x p1.x)))
		   (b (- p1.y
			 (* m p1.x)))
		   (inliers (std--vector<Point2D>)))
	       (foreach (p data)
			(when (< (distance p m b)
				 RANSAC_INLIER_THRESHOLD)
			  (inliers.push_back p)))
					;,(lprint :vars `(idx1 idx2 (data.size) (inliers.size) m b))
	       (when (< RANSAC_MIN_INLIERS 
			(inliers.size))
		 (let ((sum_x 0d0)
		       (sum_y 0d0))
		   (foreach (p inliers)
			    (incf sum_x p.x)
			    (incf sum_y p.y))
		   (let ((avg_x (/ sum_x (inliers.size)))
			 (avg_y (/ sum_y (inliers.size)))
			 (var_x 0d0)
			 (cov_xy 0d0))
		     (foreach (p inliers)
			      (incf var_x (* (- p.x avg_x)
					     (- p.x avg_x)))

			      (incf cov_xy (* (- p.x avg_x)
					      (- p.y avg_y))))
		     (let ((m (/ cov_xy var_x))
			   (b (- avg_y (* m avg_x))))
					;,(lprint :msg "stat" :vars `(m b))
		       (when (< (best_inliers.size)
				(inliers.size))
			 (setf best_inliers inliers
			       best_m m
			       best_b b))))))
	       )))
	 (setf m best_m
	       b best_b
	       inliers best_inliers)
	 ))

     (defun main
	 (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (let ((m0 1.0d0)
	     (b0 2.0d0)
	     (noise_stddev .1d0))
					;,(lprint :vars `(m0 b0))
	 (let ((seed (dot (std--chrono--system_clock--now)
			  (time_since_epoch)
			  (count))))
	   "std::default_random_engine generator(seed);")
	 "std::normal_distribution<double> distribution(0.0,noise_stddev);"
	 )
       (let ((count 0))
	(while true
	       
	       (do0
		(incf count)
		(do0			; dotimes (i N_FIFO)
		 (let ((x (/ (* 1.0 count) N_FIFO))
		       (p (dot (Line m0 b0)
			       (point x))))
		   (incf p.y (distribution generator))
		   (when (< (- N_FIFO 1) (fifo.size))
		     (fifo.pop_back))
		   (fifo.push_front p)))
	       
		(let ((m 0d0)
		      (b 0d0)
		      (inliers ("std::vector<Point2D>")))
		  (ransac_line_fit fifo m b inliers)
					;,(lprint :vars `(m b))
		  (do0
		   (let ((X (std--vector<double>))
			 (X3 (std--vector<double>))
			 (Y3 (std--vector<double>))
			 (Y0 (std--vector<double>))
			 (Y1 (std--vector<double>))
			 (Y2 (std--vector<double>)))
		     (dotimes (i (fifo.size))
		       (let ((x (dot (aref fifo i) x))
			     (p (dot (Line m0 b0)
				     (point x))))
			 (X.push_back x)
			 (Y0.push_back (dot (aref fifo i) y))
			 (Y1.push_back (dot (Line m b)
					    (point x)
					    y))
			 (Y2.push_back (dot p y))
			 ))
		     (foreach (p inliers)
			      (X3.push_back p.x)
			      (Y3.push_back p.y))
		     (do0
		      (plt--clf)
		      ,@(loop for e in `(Y0 Y1 Y2)
			      collect
			      `(plt--named_plot (string ,e) X ,e))
		      (plt--scatter X3 Y3 10.0 (curly (curly (string "color") (string "r"))))
		      ;;(plt--show)
		      (plt--pause .01))
		     ))))))))))



