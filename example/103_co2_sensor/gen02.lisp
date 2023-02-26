
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir*
    ; "/home/martin/src/my_fancy_app_name/main/"
    #P"example/103_co2_sensor/source02/"
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
		cmath)

     (include "core.h")

     

     ,@(let ((n-fifo 12))
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


     (defun ransac_line_fit (data m b)
       (declare (type "std::deque<Point2D>&" data)
		(type double& m b))
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
		       ,(lprint :msg "stat" :vars `(m b))
		       (when (< (best_inliers.size)
				(inliers.size))
			 (setf best_inliers inliers
			       best_m m
			       best_b b))))))
	       )))
	 (setf m best_m
	       b best_b)
	 ))

     (defun main
       (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (let ((m0 .1d0)
	     (b0 23d0))
	 ,(lprint :vars `(m0 b0))
	 (dotimes (i N_FIFO)
	   (let ((x (* 1.0 i))
		 (y (+ b0 (* m0 x)))
		 (p (Point2D (designated-initializer :x x
						     :y y))))
	     (when (< (- N_FIFO 1) (fifo.size))
	       (fifo.pop_back))
	     (fifo.push_front p))))
       (dotimes (i (fifo.size))
		,(lprint :vars `(i
				 (dot (aref fifo i) x)
				 (dot (aref fifo i) y))))
       (let ((m 0d0)
	     (b 0d0))
	 (ransac_line_fit fifo m b)
	 ,(lprint :vars `(m b))))
     

     )))


