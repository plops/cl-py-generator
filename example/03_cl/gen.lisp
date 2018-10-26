(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))

;; https://www.youtube.com/watch?v=Zz_6P5qAJck
(in-package :cl-py-generator)

(start-python)

(let ((code
       `(do0
	 (imports (sys
		   os
                   (lp loopy)
                   (cl pyopencl)
                   (np numpy)
                   pyopencl.array))
         (setf X (dot (np.random.random (tuple 7000 3))
                      (astype np.float32)))
         (setf ctx (cl.create_some_context :interactive False)
               q (cl.CommandQueue ctx)
               X_dev (cl.array.to_device q X)
               )
         (setf knl (lp.make_kernel (string
                                    "{[i,j,k]:0<=i,j<M and 0<=k<N}"
                                    #+nil,(emit-py :code `(dict ((list i j k) (and (<= 0 (tuple i j) (- M 1))
                                                                                    (<= 0 k (- N 1)))))))
                                   (string ,(emit-py :code `(= (aref D i j)
                                                                  (sqrt (sum k (** (- (aref X i k)
                                                                                      (aref X j k))
                                                                                   2))))))
                                   :lang_version (tuple 2018 2))
               knl (lp.set_options knl :write_cl True)
               knl (lp.set_loop_priority knl (string "i,j"))
               result (knl q :X X_dev))
	 )))
  ;(run code)
  (write-source "/home/martin/stage/cl-py-generator/example/03_cl/source/code" code))


