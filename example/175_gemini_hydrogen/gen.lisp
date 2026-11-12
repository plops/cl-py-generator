(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))

(in-package :cl-py-generator)

(let ((code
       `(do0
         "# Automatisch generiert mit cl-py-generator"
         (imports ((np numpy)
                   (plt matplotlib.pyplot)
                   (sig scipy.signal)
                   (ndi scipy.ndimage)))

         "# 1. Verrauschtes Gauss-Signal + exakte Ableitung"
         (setf rng (np.random.default_rng 42)
               x (np.linspace -5.0 5.0 501)
               dx (- (aref x 1) (aref x 0))
               sigma_sig 1.0
               noise_std 0.05
               y_clean (np.exp (* -0.5 (np.square (/ x sigma_sig))))
               dy_true (* (/ (* -1.0 x) (np.square sigma_sig)) y_clean)
               y_noisy (+ y_clean (rng.normal 0.0 noise_std (len x))))

         "# 2x2 Subplots + Unpacking via ,@(loop ...)"
         (setf (ntuple fig axes) (plt.subplots 2 2 :figsize (tuple 13 8.5))
               ,@(loop for i below 2 append
                       (loop for j below 2 append
                             `(,(intern (format nil "AX~d~d" i j))
                               (aref (aref axes ,i) ,j)))))

	 "# 2. SavGol-Sweep (Bias-Varianz-Tradeoff)"
         (setf windows (np.arange 5 103 2)
               best_rmse np.inf
               best_win 5
               best_poly 1)

         (for (p (list 1 3 5))
           (do0
             (setf errs (list)
                   valid_wins (aref windows (> windows p)))
             (for (w valid_wins)
               (do0
                 (setf dy_tmp (sig.savgol_filter y_noisy
                                                 :window_length (int w)
                                                 :polyorder p :deriv 1 :delta dx)
                       rmse (np.sqrt (np.mean (np.square (- dy_tmp dy_true)))))
                 (errs.append rmse)
                 (if (< rmse best_rmse)
                   (setf best_rmse rmse
                         best_win (int w)
                         best_poly p))))
             (ax10.plot valid_wins errs :label (dot (string "SavGol poly={}") (format p)))))

         (setf dy_savgol (sig.savgol_filter y_noisy
                                            :window_length best_win
                                            :polyorder best_poly :deriv 1 :delta dx))

         "# 3. Alternativen: finite Differenz & optimierter Gauss-Ableitungsfilter"
         (setf dy_grad (np.gradient y_noisy dx)
               rmse_grad (np.sqrt (np.mean (np.square (- dy_grad dy_true))))
               best_rmse_g np.inf
               best_sigma_g 0.5)

         (for (sg (np.linspace 0.05 1.5 60))
           (do0
             (setf dy_g (/ (ndi.gaussian_filter1d y_noisy :sigma (/ sg dx) :order 1) dx)
                   rmse_g (np.sqrt (np.mean (np.square (- dy_g dy_true)))))
             (if (< rmse_g best_rmse_g)
               (setf best_rmse_g rmse_g
                     best_sigma_g sg))))

         (setf dy_gauss (/ (ndi.gaussian_filter1d y_noisy :sigma (/ best_sigma_g dx) :order 1) dx))

         "# 4. Filter-Eigenschaften: Impulsantworten, Noise-Gain, Momente & FFT"
         (setf imp (np.zeros 201)
               (aref imp 100) 1.0
               h_sg (sig.savgol_coeffs best_win best_poly :deriv 1 :delta dx)
               h_gauss (/ (ndi.gaussian_filter1d imp :sigma (/ best_sigma_g dx) :order 1) dx)
               h_grad (np.array (list (/ -0.5 dx) 0.0 (/ 0.5 dx)))
               half_w (int (/ (- best_win 1) 2))
               k_grid (np.arange (* -1 half_w) (+ half_w 1))
               dc_sg (np.sum h_sg)
               slope_sg (np.sum (* (* -1.0 k_grid) dx h_sg))
               n_fft 4096
               freqs (np.fft.rfftfreq n_fft :d dx)
               amp_ideal (* 2.0 np.pi freqs)
               ,@(loop for name in '(sg gauss grad) append
                       (let ((h (intern (string-upcase (format nil "h_~a" name))))
                             (ng (intern (string-upcase (format nil "ng_~a" name))))
                             (amp (intern (string-upcase (format nil "amp_~a" name)))))
                         `(,ng (np.sqrt (np.sum (np.square ,h)))
                           ,amp (np.abs (np.fft.rfft ,h :n n_fft)))))
               mask_pos (> freqs 0.0)
               rel_sg (/ (aref amp_sg mask_pos) (aref amp_ideal mask_pos))
               f_3db_sg (aref (aref freqs mask_pos)
                              (np.argmax (< rel_sg (/ 1.0 (np.sqrt 2.0))))))

         (print (dot (string "RMSE      -> SavGol(w={},p={}): {:.5f} | Gauss(s={:.3f}): {:.5f} | Grad: {:.5f}")
                     (format best_win best_poly best_rmse best_sigma_g best_rmse_g rmse_grad)))
         (print (dot (string "NoiseGain -> SavGol: {:.3f} | Gauss: {:.3f} | Grad: {:.3f}")
                     (format ng_sg ng_gauss ng_grad)))
         (print (dot (string "SavGol    -> DC-Summe: {:.2e} | Rampen-Steigung: {:.6f} | f_-3dB: {:.2f} Hz")
                     (format dc_sg slope_sg f_3db_sg)))

         "# 5. Plots"
         (ax00.plot x y_noisy :label (string "Verrauscht") :alpha 0.5)
         (ax00.plot x y_clean :label (string "Sauberer Gauss") :color (string "k") :linewidth 2)

         (ax01.plot x dy_grad :label (string "np.gradient") :alpha 0.35 :color (string "gray"))
         (ax01.plot x dy_true :label (string "Exakt g'(x)") :color (string "k") :linewidth 2)
         (ax01.plot x dy_savgol :label (dot (string "SavGol (w={}, p={})") (format best_win best_poly)) :linewidth 2)
         (ax01.plot x dy_gauss :label (dot (string "Gauss-Deriv (s={:.2f})") (format best_sigma_g))
                    :linestyle (string "--") :linewidth 2)

         (ax10.axvline best_win :color (string "r") :linestyle (string ":")
                       :label (dot (string "Optimum w={}") (format best_win)))

         (setf f_mask (<= freqs 6.0))
         (ax11.plot (aref freqs f_mask) (aref amp_ideal f_mask)
                    :label (string "Ideal: 2*pi*f") :color (string "k") :linestyle (string ":"))
         ,@(loop for (amp lbl style) in
                 '((amp_grad "Zentrale Diff." "-")
                   (amp_sg "SavGol Opt." "-")
                   (amp_gauss "Gauss-Deriv Opt." "--"))
                 collect `(ax11.plot (aref freqs f_mask) (aref ,amp f_mask)
                                     :label (string ,lbl) :linestyle (string ,style)))
         (ax11.axvline f_3db_sg :color (string "r") :linestyle (string ":")
                       :label (dot (string "SavGol -3dB ({:.2f} Hz)") (format f_3db_sg)))

         ,@(loop for (ax title xlab ylab) in
                 '((ax00 "Eingangssignal" "x" "y")
                   (ax01 "Vergleich der 1. Ableitung" "x" "dy/dx")
                   (ax10 "SavGol Optimierung (Bias-Varianz)" "window_length" "RMSE vs. exakt")
                   (ax11 "Frequenzgang |H(f)|" "Frequenz [1/x]" "|H(f)|"))
                 append `((dot ,ax (set_title (string ,title)))
                          (dot ,ax (set_xlabel (string ,xlab)))
                          (dot ,ax (set_ylabel (string ,ylab)))
                          (dot ,ax (legend))
                          (dot ,ax (grid :alpha 0.3))))

         (plt.tight_layout)
         (plt.savefig (string "savgol_analyse.png") :dpi 150)
         (plt.show))))
  (write-source "/home/kiel/stage/cl-py-generator/example/175_gemini_hydrogen/source/savgol_test" code))
