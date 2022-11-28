(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (assert (eq :invert
	      (readtable-case *readtable*)))
  (defparameter *repo-dir-on-host* "/home/martin/stage/cl-py-generator")
  (defparameter *repo-dir-on-github* "https://github.com/plops/cl-py-generator/tree/master/")
  (defparameter *example-subdir* "example/84_lte")
  (defparameter *path* (format nil "~a/~a" *repo-dir-on-host* *example-subdir*) )
  (defun lprint (&key (msg "") vars)
    `(print (dot (string ,(format nil "{:7.6f} \\033[31m ~a \\033[0m ~{~a={}~^ ~}" msg vars))
                 (format
		  (- (time.time) start_time)
                  ,@vars))))
  (let ((nb-counter 1))
    (flet ((gen (path code)
	     "create jupyter notebook file in a directory below source/"
	     (let ((fn  (format nil "source/~3,'0d_~{~a~^_~}.ipynb" nb-counter path)))
	       (write-notebook
		:nb-file fn
		:nb-code (append `((python (do0
					    (comments
					     ,(format nil "default_exp ~{~a~^/~}_~2,'0d" path nb-counter)))))
				 code))
	       (format t "~&~c[31m wrote Python ~c[0m ~a~%"
		       #\ESC #\ESC fn))
	     (incf nb-counter)))
      (let* ()
	(gen `(lte)
	     `((python
		(cell
		 (imports ((plt matplotlib.pyplot)))
		 (plt.ion)
		 (imports (pathlib
			   time
			   tqdm
			   (pd pandas)
			   (np numpy)
			   tqdm
			   ))
		 (imports-from (matplotlib.pyplot plot figure scatter gca sca subplots subplots_adjust title xlabel ylabel xlim ylim grid))
		 (setf start_time (time.time))))

	       (python
		(cell
		 ,@(loop for (e f) in `((font.size 6)
					(figure.facecolor (string "w"))
					(figure.figsize (tuple 10 5)))
			 collect
			 `(setf (aref plt.rcParams (string ,e))
				,f))
		 ))
	       (python
		(cell
		 (comments "30.72 Msps")
		 (setf fs 30_720_000)
		 (comments "extract PUSCH slots")
		 (setf t0 .0757
		       T0 10e-3
		       x (np.fromfile (string "LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-data")
				      (string "int16")
				      :offset (int (* t0 fs 4))
				      :count (int (* T0 fs 2)))

		       )
		 (comments "scale amplitude so that PUSCH QPSK symbols are +/-1, +/-1j")
		 (setf x (/ (+ (aref x (slice "" "" 2))
			       (* 1j
				  (aref x (slice 1  "" 2))))
			    2e4))
		 (comments "no carrier at DC, central carriers at +/-7.5kHz"
			   "shift up by 5.94kHz to move left central subcarrier to DC. "
			   "carrier frequency offset measured with PUSCH and DMRS to be ~1.5kHz."
			   "phase offset with DMRS signal")
		 (setf delta_f 5.94e3
		       delta_phi 1.96
		       x (* x (np.exp
			       (* 1j (+ (*
					 2
					 np.pi
					 (/ delta_f fs)
					 (np.arange x.size))
					delta_phi)))))
		 ,(lprint :msg "carrier frequency offset"
			  :vars `((- delta_f 7.5e3)))
		 ,@(loop for e in
			 `((Tu 2048 "useful time of a symbol")
			   (cp_len 144 "4.7usec, used in symbols 1-6 in a slot")
			   (cp_len2 160 "5.2usec, used in symbol 0 in a slot"))
			 collect
			 (destructuring-bind (name val comment) e
			   `(do0
			     (setf ,name ,val)
			     ,(lprint :msg (format nil "~a" comment)
				      :vars `(,name)))))))
	       (python
		(cell

		 (setf cp_corr (np.empty (- x.size
					    cp_len
					    Tu)
					 np.complex64))
		 (for (j (tqdm.tqdm (range cp_corr.size)))
		      (setf (aref cp_corr j)
			    (np.sum
			     (*
			      (aref (aref x (slice j ""))
				    (slice "" cp_len))
			      (np.conjugate
			       (aref (aref x (slice (+ j Tu) ""))
				     (slice "" cp_len)))))))
		 ,(lprint
		   :msg "correlate the end of each symbol with its own cyclic prefix"
		   )
		 (do0
		  (setf L (* 20 Tu)
			xsample (np.arange L)
			xtime (* 1e6 (/ xsample
					fs)))
		  ,(lprint
		    :vars `((len cp_corr) L))
		  ,(let ((def-plot `((:name I :fun np.real)
				     (:name Q :fun np.imag)
				     (:name abs :fun np.abs)
				     (:name angle :fun np.angle))))
		     `(do0
		       (setf (ntuple fig axs) (subplots ,(length def-plot) 1
							:figsize (list 12 17)))
		       (subplots_adjust :left .1
					:right .9
					:bottom .05
					:top .95
					:wspace .3
					:hspace .4)
		       ,@(loop for e in def-plot
			       and e-i from 0
			       collect
			       (destructuring-bind (&key name fun ) e
				 `(do0
				   (setf ax (aref axs ,e-i))
				   (plt.sca ax)

				   (plot xsample (,fun (aref cp_corr (slice "" L)))
					 :label (string ,name))
				   (do0
				    (comments "upper axis function of lower axis"
					      "https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib")
				    (setf ax2 (dot ax (twiny)))
				    (ax2.set_xticks (ax.get_xticks))
				    (ax2.set_xbound (ax.get_xbound))
				    (ax2.set_xticklabels
				     (list
				      (for-generator ( x (ax.get_xticks))
						     (* 1e6 (/ x fs))))))

				   (ax2.set_xlabel (string "time (us)"))
				   (ax.set_xlabel (string "time (sample)"))
					; (axes2.set_xlabel (string "samples"))
				   (ylabel (string ,name))
				   (grid)
				   (plt.legend)
				   )))))
		  ))
		))))))
  (sb-ext:run-program "/usr/bin/sh"
		      `("/home/martin/stage/cl-py-generator/example/84_lte/source/setup01_nbdev.sh"))
  (format t "~&~c[31m ran nbdev ~c[0m ~%" #\ESC #\ESC ))




