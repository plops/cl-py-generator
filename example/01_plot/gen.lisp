(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))
;; https://www.kdnuggets.com/2018/08/introduction-t-sne-python.html
;; https://www.youtube.com/watch?v=RJVL80Gg3lA
;; sudo pacman -S autopep8

(in-package :cl-py-generator)

(start-python)

(let ((code
       `(do0
	 (imports (sys
		   (plt matplotlib.pyplot)
		   (np numpy)
		   (pd pandas)
		   pathlib))
	 (plt.ion)
	 (setf x (np.linspace 0 2.0 30)
	       y (np.sin x))
	 (plt.plot x y)
	 (plt.grid))))
  (run code)
  (write-source "/home/martin/stage/cl-py-generator/example/01_plot/source/code" code))
