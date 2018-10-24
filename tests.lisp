
(emit-py :code
	 `(def gf (a &key (sigma 1))
	    (setf r a.copy
		  (dot r values) (scipy.ndimage.filters.gaussian_filter a :sigma sigma)
		  r.name (string gauss)
		  )
	    (return r)))

(write-source "/dev/shm/o"
	      `(do0
		(imports (sys
			  (plt matplotlib.pyplot)
			  (np numpy)
			  (xr xarray)
			  (xrp xarray.plot)))
		(plt.ion)
		(def gf (a &key (sigma 1))
		  (setf r a.copy
			(dot r values) (scipy.ndimage.filters.gaussian_filter a :sigma sigma)
			r.name (string gauss)
			)
		  (return r))
		(try
		 (setf x (np.linspace 0 3 30)
		       y (np.sin x)
		       (aref y (slice None -1 1)) (np.cos (aref x (slice None -1 1)))))
		(plt.plot x y)))

(emit-py :code
	 `(imports ((plt matplotlib.pyplot)
		    (np numpy)
		    (xr xarray)
		    (xrp xarray.plot))))



(emit-py :code
	 `(if (== a b)
	      (do0 3
		  (setf 3 34
			tnt iena))
	      4))

(emit-py :code
	 `(for (i (np.linspace 3 3))
	       (= twf en)
	       (setf taen ea)
	       (setf q 3
		     t f
		     atrs l)
	       (for (j (np.linspace 3 :end 3))
		    (= a (+ i j))
		    (= b (+ i j))
		    (= c (+ i j))
		    break)))

(emit-py :code
	 `(for ((list a b) (enumerate (range (+ c 10))))
	       (for (a (range 3))
		    (setf q (+ 33d3 a9 (+ 32 3))
			  a (// b e)
			  c (/ d 3)
			  e f)
		    (setf qt (or True True False)))
	       (- a ,(complex 3.3 3))))

(emit-py :code
	 `(do (list 1 2 3)
	      (tuple 1 2 3)))

(emit-py :code
	 `(def alph (a b c &key (ar 3))
	    (for ((list a b) (enumerate (range (+ c 10))))
		 (setf q (+ 33 a9 (+ 32 3))
		       qt (or True True False)
		       tain (+ q qt)))
	    (return (* 3 qt))))
(emit-py :code
	 `(def alph (a b c &key (ar 3))
	    (setf q (+ 33 9 9 (+ 32 3))
		  qt (or True True False)
		  ast (- at f))))
(emit-py :code
	 `(def alph (a b c)
	    (list 1 2 3)))

(emit-py :code
	 `(do (do (list (tuple 1 2) (tuple 1 3) (tuple 1 2 3)))))

(emit-py :code
	 `(do 1 2 (list 3 3)))

(emit-py :code
	 `(list 1 2 3))
(emit-py :code
	 `(tuple 1 2 3))

(emit-py :code
	 `(complex 1d0 0))
(emit-py :code
	 `123)
(emit-py :code
	 `"bla")

(emit-py :code
	 `(string "bla"))

(emit-py :code
	 `(string "bla"))
