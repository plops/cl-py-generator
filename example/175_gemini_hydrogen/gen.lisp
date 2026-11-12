(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-py-generator))

(in-package :cl-py-generator)

(defun pyfmt (fmt &rest args)
  `(dot (string ,fmt) (format ,@args)))

(defun rmse-stats (pred target)
  `(np.sqrt (np.mean (np.square (- ,pred ,target)) :axis -1)))

(defun setup-grid (rows cols figsize)
  `(setf (ntuple fig axes) (plt.subplots ,rows ,cols :figsize (tuple ,@figsize))
         ,@(loop for i below rows append
                 (loop for j below cols append
                       `(,(intern (format nil "AX~d~d" i j))
                         (aref (aref axes ,i) ,j))))))

(defun decorate-ax (ax title xlab ylab &key yscale xlim ylim)
  `(do0
    (dot ,ax (set_title (string ,title)))
    (dot ,ax (set_xlabel (string ,xlab)))
    (dot ,ax (set_ylabel (string ,ylab)))
    ,@(when yscale `((dot ,ax (set_yscale (string ,yscale)))))
    ,@(when xlim `((dot ,ax (set_xlim ,@xlim))))
    ,@(when ylim `((dot ,ax (set_ylim ,@ylim))))
    (dot ,ax (legend :fontsize 8.5))
    (dot ,ax (grid :alpha 0.3 :which (string "both")))))

(let ((code
       `(do0
         "# Automatisch generiert mit cl-py-generator"
         (imports ((np numpy)
                   (plt matplotlib.pyplot)
                   (sig scipy.signal)
                   (ndi scipy.ndimage)))

         "# 1. Signal + Monte-Carlo-Ensemble (n_mc x n_x)"
         (setf rng (np.random.default_rng 42)
               x (np.linspace -5.0 5.0 501)
               dx (- (aref x 1) (aref x 0))
               sigma_sig 1.0
               noise_std 0.05
               n_mc 40
               y_clean (np.exp (* -0.5 (np.square (/ x sigma_sig))))
               dy_true (* (/ (* -1.0 x) (np.square sigma_sig)) y_clean)
               y_noisy_mc (+ y_clean (rng.normal 0.0 noise_std (tuple n_mc (len x)))))

         ,(setup-grid 2 2 '(13.5 9.0))

         "# 2. SavGol Monte-Carlo-Sweep (w = 5..251)"
         (setf windows (np.arange 5 253 2)
               best_rmse np.inf
               best_win 5
               best_poly 1)

         (for (p (list 1 3 5))
           (do0
             (setf means (list)
                   stds (list)
                   valid_wins (aref windows (> windows p)))
             (for (w valid_wins)
               (do0
                 (setf dy_tmp (sig.savgol_filter y_noisy_mc
                                                 :window_length (int w)
                                                 :polyorder p :deriv 1 :delta dx :axis -1)
                       rmses ,(rmse-stats 'dy_tmp 'dy_true)
                       m (np.mean rmses)
                       s (np.std rmses))
                 (means.append m)
                 (stds.append s)
                 (if (< m best_rmse)
                   (setf best_rmse m
                         best_win (int w)
                         best_poly p))))
             (setf m_arr (np.array means)
                   s_arr (np.array stds))
             (ax10.plot valid_wins m_arr :label ,(pyfmt "SavGol poly={}" 'p))
             (ax10.fill_between valid_wins
                                (np.maximum 1e-6 (- m_arr s_arr))
                                (+ m_arr s_arr) :alpha 0.18)))

         (setf dy_savgol (sig.savgol_filter y_noisy_mc
                                            :window_length best_win
                                            :polyorder best_poly :deriv 1 :delta dx :axis -1))

         "# 3. Alternativen (Gradient & Gauss-Ableitungsfilter)"
         (setf dy_grad (np.gradient y_noisy_mc dx :axis -1)
               rmse_grad (np.mean ,(rmse-stats 'dy_grad 'dy_true))
               best_rmse_g np.inf
               best_sigma_g 0.5)

         (for (sg (np.linspace 0.05 1.5 60))
           (do0
             (setf dy_g (/ (ndi.gaussian_filter1d y_noisy_mc :sigma (/ sg dx) :order 1 :axis -1) dx)
                   m_g (np.mean ,(rmse-stats 'dy_g 'dy_true)))
             (if (< m_g best_rmse_g)
               (setf best_rmse_g m_g
                     best_sigma_g sg))))

         (setf dy_gauss (/ (ndi.gaussian_filter1d y_noisy_mc :sigma (/ best_sigma_g dx) :order 1 :axis -1) dx))

         "# 4. Filter-Eigenschaften, Bias-Varianz & 2Hz-Stoertest"
         (setf imp (np.zeros 1001)
               (aref imp 500) 1.0
               h_sg (sig.savgol_coeffs best_win best_poly :deriv 1 :delta dx)
               h_gauss (/ (ndi.gaussian_filter1d imp :sigma (/ best_sigma_g dx) :order 1) dx)
               h_grad (np.array (list (/ -0.5 dx) 0.0 (/ 0.5 dx)))
               n_fft 4096
               freqs (np.fft.rfftfreq n_fft :d dx)
               amp_ideal (* 2.0 np.pi freqs)
               ,@(loop for name in '(sg gauss grad) append
                       (let ((h (intern (string-upcase (format nil "h_~a" name))))
                             (ng (intern (string-upcase (format nil "ng_~a" name))))
                             (amp (intern (string-upcase (format nil "amp_~a" name)))))
                         `(,ng (np.sqrt (np.sum (np.square ,h)))
                           ,amp (np.abs (np.fft.rfft ,h :n n_fft)))))
               ,@(loop for (name r ng) in '((sg best_rmse ng_sg)
                                            (gauss best_rmse_g ng_gauss))
                       append
                       (let ((ns (intern (string-upcase (format nil "noise_~a" name))))
                             (bs (intern (string-upcase (format nil "bias_~a" name)))))
                         `(,ns (* ,ng noise_std)
                           ,bs (np.sqrt (np.maximum 0.0 (- (np.square ,r) (np.square ,ns)))))))
               mask_pos (> freqs 0.0)
               rel_sg (/ (aref amp_sg mask_pos) (aref amp_ideal mask_pos))
               f_3db_sg (aref (aref freqs mask_pos)
                              (np.argmax (< rel_sg (/ 1.0 (np.sqrt 2.0)))))
               y_tone (+ y_clean (* 0.05 (np.sin (* 2.0 np.pi 2.0 x))))
               dy_tone_sg (sig.savgol_filter y_tone :window_length best_win
                                             :polyorder best_poly :deriv 1 :delta dx)
               dy_tone_g (/ (ndi.gaussian_filter1d y_tone :sigma (/ best_sigma_g dx) :order 1) dx)
               rmse_tone_sg ,(rmse-stats 'dy_tone_sg 'dy_true)
               rmse_tone_g ,(rmse-stats 'dy_tone_g 'dy_true))

         ,@(loop for (fmt args) in
                 '(("MC-RMSE (n={}) -> SavGol(w={},p={}): {:.5f} | Gauss(s={:.3f}): {:.5f} | Grad: {:.5f}"
                    (n_mc best_win best_poly best_rmse best_sigma_g best_rmse_g rmse_grad))
                   ("Bias / Noise     -> SavGol: {:.5f} / {:.5f} | Gauss: {:.5f} / {:.5f} | f_-3dB: {:.2f} Hz"
                    (bias_sg noise_sg bias_gauss noise_gauss f_3db_sg))
                   ("2.0Hz-Stoertest  -> SavGol RMSE: {:.5f} | Gauss RMSE: {:.5f}"
                    (rmse_tone_sg rmse_tone_g)))
                 collect `(print ,(apply #'pyfmt fmt args)))

         "# 5. Plots"
         (ax00.plot x (aref y_noisy_mc 0) :label (string "Verrauscht (Seed 0)") :alpha 0.5)
         (ax00.plot x y_clean :label (string "Sauberer Gauss") :color (string "k") :linewidth 2)

         (ax01.plot x (aref dy_grad 0) :label (string "np.gradient") :alpha 0.3 :color (string "gray"))
         (ax01.plot x dy_true :label (string "Exakt g'(x)") :color (string "k") :linewidth 2)
         (ax01.plot x (aref dy_savgol 0) :label ,(pyfmt "SavGol (w={}, p={})" 'best_win 'best_poly) :linewidth 2)
         (ax01.plot x (aref dy_gauss 0) :label ,(pyfmt "Gauss-Deriv (s={:.2f})" 'best_sigma_g)
                    :linestyle (string "--") :linewidth 2)

         (ax10.axvline best_win :color (string "r") :linestyle (string ":")
                       :label ,(pyfmt "SavGol Opt. w={}" 'best_win))
         (ax10.axhline best_rmse_g :color (string "k") :linestyle (string "--")
                       :label ,(pyfmt "Gauss Opt. ({:.4f})" 'best_rmse_g))

         (setf f_mask (<= freqs 3.0)
               d_spec (* freqs (np.exp (* -2.0 (np.square (* np.pi sigma_sig freqs)))))
               d_spec (* 2.5 (/ d_spec (np.max d_spec))))
         (ax11.fill_between (aref freqs f_mask) 0.0 (aref d_spec f_mask)
                            :color (string "0.85") :alpha 0.6 :label (string "Spektrum |g'(f)| (norm.)"))
         (ax11.plot (aref freqs f_mask) (aref amp_ideal f_mask)
                    :label (string "Ideal: 2*pi*f") :color (string "k") :linestyle (string ":"))
         ,@(loop for (amp lbl style) in
                 '((amp_grad "Zentrale Diff." "-")
                   (amp_sg "SavGol Opt." "-")
                   (amp_gauss "Gauss-Deriv Opt." "--"))
                 collect `(ax11.plot (aref freqs f_mask) (aref ,amp f_mask)
                                     :label (string ,lbl) :linestyle (string ,style)))
         (ax11.axvline f_3db_sg :color (string "r") :linestyle (string ":")
                       :label ,(pyfmt "SavGol -3dB ({:.2f} Hz)" 'f_3db_sg))

         ,@(loop for spec in
                 '((ax00 "Eingangssignal" "x" "y")
                   (ax01 "Vergleich der 1. Ableitung (Realisierung 0)" "x" "dy/dx")
                   (ax10 "SavGol MC-Optimierung (Mean +/- 1 std)" "window_length" "RMSE vs. exakt"
                         :yscale "log")
                   (ax11 "Frequenzgang |H(f)| vs. Signalspektrum" "Frequenz [1/x]" "|H(f)|"
                         :xlim (0.0 3.0) :ylim (0.0 5.5)))
                 collect (apply #'decorate-ax spec))

         (plt.tight_layout)
         (plt.savefig (string "savgol_analyse.png") :dpi 150)
         (plt.show))))
  (write-source "/home/kiel/stage/cl-py-generator/example/175_gemini_hydrogen/source/savgol_test" code))
