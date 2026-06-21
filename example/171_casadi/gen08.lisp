(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator))

(in-package #:g)

;; ============================================================================
;; LISP HELPER FUNCTIONS (S-EXPRESSION MACROS / TEMPLATES)
;; ============================================================================

;; Helper function to generate stage cost (quadratic cost terms)
(defun stage-cost (xk uk q1 q2 q3 q4 r)
  `(+ (* ,q1 (** (aref ,xk 0) 2))
      (* ,q2 (** (aref ,xk 1) 2))
      (* ,q3 (** (aref ,xk 2) 2))
      (* ,q4 (** (aref ,xk 3) 2))
      (* ,r (** ,uk 2))))

;; Helper function to generate discrete Euler dynamics update
(defun discrete-dynamics (xk uk vk A B G dt)
  `(+ ,xk (* ,dt (+ (@ ,A ,xk) (@ ,B ,uk) (* ,G ,vk)))))

(progn
  (defparameter *source* "example/171_casadi/")

  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p08_active_suspension"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot))))

     (comments
      "========================================================================"
      "ACTIVE SUSPENSION CONTROL USING LINEAR MODEL PREDICTIVE CONTROL (MPC)"
      "========================================================================"
      "Topic: Quarter-Car Active Suspension System Model Predictive Control"
      ""
      "Why it is interesting:"
      "Active suspension systems improve ride comfort (reducing passenger"
      "chassis acceleration) and road holding (maintaining tire contact) by dynamically"
      "adjusting actuator forces between the chassis and wheel assembly."
      "It represents a classic engineering trade-off: comfort vs. handling."
      ""
      "Why QP is advantageous here:"
      "1. Hard Physical Constraints: Actuators have force limits (saturation),"
      "   and the suspension has physical travel limits (stroke bounds)."
      "   Linear controllers (like LQR or PID) cannot handle these constraints"
      "   optimally and safely."
      "2. Predictive Action: By solving a Quadratic Program at each step,"
      "   the controller 'looks ahead' at future road disturbances (if known or"
      "   estimated) and pre-emptively adjusts the suspension force."
      "3. Speed: A QP solver is extremely fast and guarantees finding the global"
      "   optimum, making real-time execution in vehicle hardware feasible."
      "========================================================================")

     ;; Model Parameters
     (setf ms 300.0   ;; Chassis mass (kg)
	   mu 40.0    ;; Wheel assembly mass (kg)
	   ks 15000.0 ;; Suspension spring rate (N/m)
	   cs 1000.0  ;; Passive damper rate (N-s/m)
	   kt 150000.0 ;; Tire spring rate (N/m)
	   )

     (setf dt 0.01   ;; Discretization time-step (s)
	   N 30      ;; Prediction horizon steps (0.3 seconds lookahead)
	   )

     ;; Continuous dynamics matrices (Quarter-car model)
     (setf A_np (np.array (list (list 0.0 1.0 0.0 -1.0)
				(list (/ (- ks) ms) (/ (- cs) ms) 0.0 (/ cs ms))
				(list 0.0 0.0 0.0 1.0)
				(list (/ ks mu) (/ cs mu) (/ (- kt) mu) (/ (- cs) mu))))
	   B_np (np.array (list 0.0 (/ 1.0 ms) 0.0 (/ -1.0 mu)))
	   G_np (np.array (list 0.0 0.0 -1.0 0.0)))

     (setf A (DM A_np)
	   B (DM B_np)
	   G (DM G_np))

     (comments "--- MPC Weights ---")
     (setf q1 100000.0  ;; Weight on suspension deflection (comfort / travel)
	   q2 5000.0    ;; Weight on body vertical velocity (comfort)
	   q3 100000.0  ;; Weight on tire deflection (safety / road holding)
	   q4 10.0      ;; Weight on wheel velocity
	   r  0.01      ;; Weight on control input (actuator force)
	   )

     ;; Terminal weights (scaled up for stability)
     (setf q1_N (* 10.0 q1)
	   q2_N (* 10.0 q2)
	   q3_N (* 10.0 q3)
	   q4_N (* 10.0 q4))

     (comments "--- Symbolic QP formulation ---")
     ;; States X = [x_0, x_1, ..., x_N] and Inputs U = [u_0, u_1, ..., u_{N-1}]
     (setf X (list ,@(loop for i from 0 to 30 collect `(SX.sym (string ,(format nil "x_~d" i)) 4)))
	   U (list ,@(loop for i from 0 below 30 collect `(SX.sym (string ,(format nil "u_~d" i)) 1))))

     ;; Parameters: initial state (4) + road disturbance velocity over horizon (N)
     (setf p (SX.sym (string "p") (+ 4 N))
	   x_init (aref p (slice 0 4))
	   V_r (aref p (slice 4 nil)))

     (setf f 0.0
	   g (list))

     ;; Construct objective function and equality constraints
     ,@(loop for k from 0 below 30 collect
	     `(do0
	       (setf f (+ f ,(stage-cost `(aref X ,k) `(aref U ,k) 'q1 'q2 'q3 'q4 'r)))
	       (dot g (append (- (aref X ,(1+ k))
				 ,(discrete-dynamics `(aref X ,k) `(aref U ,k) `(aref V_r ,k) 'A 'B 'G 'dt))))))

     ;; Terminal cost
     (setf f (+ f (* q1_N (** (aref (aref X N) 0) 2))
		  (* q2_N (** (aref (aref X N) 1) 2))
		  (* q3_N (** (aref (aref X N) 2) 2))
		  (* q4_N (** (aref (aref X N) 3) 2))))

     ;; Initial state equality constraint
     (dot g (insert 0 (- (aref X 0) x_init)))

     ;; Formulate high-level QP structure
     (setf qp (dict ((string "x") (vertcat (space * (paren (+ X U)))))
		    ((string "p") p)
		    ((string "f") f)
		    ((string "g") (vertcat (space * g)))))

     ;; Instantiate QP solver (qpOASES is distributed with CasADi and solves QPs very fast)
     (setf S (qpsol (string "S") (string "qpoases") qp (dict ((string "printLevel") (string "none")))))

     (comments "--- Simulation Setup ---")
     (setf sim_time 3.0
	   N_steps (int (/ sim_time dt))
	   t_vec (np.linspace 0.0 sim_time N_steps))

     (comments "Create road profile (a 5cm high bump at t = 0.5s to 0.7s)")
     (setf z_r_vec (np.zeros N_steps)
	   v_r_vec (np.zeros N_steps))

     (for (j (range N_steps))
	  (setf t_curr (aref t_vec j))
	  (if (and (<= 0.5 t_curr) (<= t_curr 0.7))
	      (do0
	       (setf (aref z_r_vec j) (* 0.025 (- 1.0 (np.cos (/ (* 2.0 np.pi (- t_curr 0.5)) 0.2)))))
	       (setf (aref v_r_vec j) (* (* 0.025 (/ (* 2.0 np.pi) 0.2)) (np.sin (/ (* 2.0 np.pi (- t_curr 0.5)) 0.2)))))
	      (do0
	       (setf (aref z_r_vec j) 0.0)
	       (setf (aref v_r_vec j) 0.0))))

     (comments "Initialize histories for Active (MPC) and Passive suspension systems")
     (setf x_hist_mpc (np.zeros (tuple 4 N_steps))
	   u_hist_mpc (np.zeros N_steps)
	   x_hist_passive (np.zeros (tuple 4 N_steps)))

     (setf x_curr (np.array (list 0.0 0.0 0.0 0.0))
	   x_curr_passive (np.array (list 0.0 0.0 0.0 0.0)))

     (comments "Define bounds for optimization variables")
     ;; State bounds: x1 (suspension stroke) in [-0.08, 0.08] m
     ;; Input bounds: actuator force u in [-1500, 1500] N
     (setf lb_state (np.array (list -0.08 -10.0 -0.05 -20.0))
	   ub_state (np.array (list 0.08 10.0 0.05 20.0))
	   lb_input (np.array (list -1500.0))
	   ub_input (np.array (list 1500.0)))

     (setf lbx (np.concatenate (tuple (np.tile lb_state (+ N 1)) (np.tile lb_input N)))
	   ubx (np.concatenate (tuple (np.tile ub_state (+ N 1)) (np.tile ub_input N))))

     (setf lbg (np.zeros (* 4 (+ N 1)))
	   ubg (np.zeros (* 4 (+ N 1))))

     (setf x0_guess (np.zeros (+ (* 4 (+ N 1)) N)))

     (comments "Simulation Loop")
     (for (j (range (- N_steps 1)))
	  ;; Save current states
	  (setf (aref x_hist_mpc (slice nil nil) j) x_curr)
	  (setf (aref x_hist_passive (slice nil nil) j) x_curr_passive)

	  ;; Extract future road profile velocity for the horizon
	  (setf V_r_horiz (np.zeros N))
	  (for (k (range N))
	       (setf idx (+ j k))
	       (if (< idx N_steps)
		   (setf (aref V_r_horiz k) (aref v_r_vec idx))
		   (setf (aref V_r_horiz k) 0.0)))

	  ;; Assemble parametric vector for the QP solver
	  (setf p_val (np.concatenate (tuple x_curr V_r_horiz)))

	  ;; Solve the QP
	  (setf sol (S :x0 x0_guess :p p_val :lbx lbx :ubx ubx :lbg lbg :ubg ubg)
		x_opt (aref sol (string "x")))

	  ;; First control input is at index 4*(N+1)
	  (setf u_opt (float (aref x_opt (* 4 (+ N 1)))))
	  (setf (aref u_hist_mpc j) u_opt)

	  ;; Update states via discrete dynamics
	  (setf x_curr (+ x_curr (* dt (+ (@ A_np x_curr) (* B_np u_opt) (* G_np (aref v_r_vec j))))))
	  (setf x_curr_passive (+ x_curr_passive (* dt (+ (@ A_np x_curr_passive) (* G_np (aref v_r_vec j)))))))

     ;; Save the final state step
     (setf (aref x_hist_mpc (slice nil nil) -1) x_curr)
     (setf (aref x_hist_passive (slice nil nil) -1) x_curr_passive)

     (comments "--- Post-processing and Reconstruction of Physical Quantities ---")
     ;; Sprung mass acceleration (comfort metric)
     (setf acc_mpc (+ (* (/ (- ks) ms) (aref x_hist_mpc 0))
		      (* (/ (- cs) ms) (- (aref x_hist_mpc 1) (aref x_hist_mpc 3)))
		      (* (/ 1.0 ms) u_hist_mpc))
	   acc_passive (+ (* (/ (- ks) ms) (aref x_hist_passive 0))
			  (* (/ (- cs) ms) (- (aref x_hist_passive 1) (aref x_hist_passive 3)))))

     ;; Displacement calculation: zs = x1 + x3 + zr
     (setf zs_mpc (+ (aref x_hist_mpc 0) (aref x_hist_mpc 2) z_r_vec)
	   zs_passive (+ (aref x_hist_passive 0) (aref x_hist_passive 2) z_r_vec))

     (comments "--- Plotting Results ---")
     ;; Global plot styling adjustments for clean layout
     (plt.rcParams.update (dict ((string "font.family") (string "sans-serif"))
				((string "font.sans-serif") (list (string "DejaVu Sans") (string "Arial")))
				((string "axes.edgecolor") (string "#cccccc"))
				((string "axes.linewidth") 0.8)
				((string "grid.color") (string "#eeeeee"))
				((string "grid.linestyle") (string "-"))))

     (setf (ntuple fig axes) (plt.subplots 2 2 :figsize (tuple 14 10)))
     (dot fig (suptitle (string "Active MPC vs. Passive Suspension System Comparison") :fontsize 16 :fontweight (string "bold") :y 0.98))

     ;; Plot 1: Road Profile and Chassis Displacement
     (setf ax1 (aref axes 0 0))
     (dot ax1 (fill_between t_vec 0 z_r_vec :color (string "#e0e0e0") :alpha 0.5 :label (string "Road Profile (Bump)")))
     (dot ax1 (plot t_vec zs_passive :label (string "Passive Chassis") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax1 (plot t_vec zs_mpc :label (string "Active Chassis (MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax1 (set_title (string "Chassis Position (zs)") :fontsize 12 :fontweight (string "bold")))
     (dot ax1 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax1 (set_ylabel (string "Displacement (m)") :fontsize 10))
     (dot ax1 (grid True :alpha 0.6))
     (dot ax1 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax1 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax1 spines) (string "right")) (set_visible False))

     ;; Plot 2: Chassis Acceleration
     (setf ax2 (aref axes 0 1))
     (dot ax2 (axhspan -0.5 0.5 :color (string "#e2f0d9") :alpha 0.6 :label (string "Comfort Zone (ISO 2631)")))
     (dot ax2 (plot t_vec acc_passive :label (string "Passive") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax2 (plot t_vec acc_mpc :label (string "Active (MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax2 (set_title (string "Chassis Vertical Acceleration") :fontsize 12 :fontweight (string "bold")))
     (dot ax2 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax2 (set_ylabel (string "Acceleration (m/s^2)") :fontsize 10))
     (dot ax2 (grid True :alpha 0.6))
     (dot ax2 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax2 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax2 spines) (string "right")) (set_visible False))

     ;; Plot 3: Suspension Deflection (Stroke)
     (setf ax3 (aref axes 1 0))
     (dot ax3 (axhspan -0.08 0.08 :color (string "#f1f3f4") :alpha 0.6 :zorder 0))
     (dot ax3 (plot t_vec (aref x_hist_passive 0) :label (string "Passive") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax3 (plot t_vec (aref x_hist_mpc 0) :label (string "Active (MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax3 (axhline :y 0.08 :color (string "#cc0000") :linestyle (string ":") :lw 1.2 :label (string "Stroke Limit (+/- 8cm)")))
     (dot ax3 (axhline :y -0.08 :color (string "#cc0000") :linestyle (string ":") :lw 1.2))
     (dot ax3 (set_title (string "Suspension Deflection (Travel)") :fontsize 12 :fontweight (string "bold")))
     (dot ax3 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax3 (set_ylabel (string "Deflection (m)") :fontsize 10))
     (dot ax3 (grid True :alpha 0.6))
     (dot ax3 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax3 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax3 spines) (string "right")) (set_visible False))

     ;; Plot 4: Actuator Control Force
     (setf ax4 (aref axes 1 1))
     (dot ax4 (axhspan -1500.0 1500.0 :color (string "#f1f3f4") :alpha 0.6 :zorder 0))
     (dot ax4 (step t_vec u_hist_mpc :label (string "Active Force (MPC)") :color (string "#34a853") :lw 2 :where (string "post")))
     (dot ax4 (axhline :y 1500.0 :color (string "#cc0000") :linestyle (string ":") :lw 1.2 :label (string "Actuator Limit (+/- 1500N)")))
     (dot ax4 (axhline :y -1500.0 :color (string "#cc0000") :linestyle (string ":") :lw 1.2))
     (dot ax4 (set_title (string "Actuator Control Force") :fontsize 12 :fontweight (string "bold")))
     (dot ax4 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax4 (set_ylabel (string "Force (N)") :fontsize 10))
     (dot ax4 (grid True :alpha 0.6))
     (dot ax4 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax4 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax4 spines) (string "right")) (set_visible False))

     (plt.tight_layout)
     (plt.savefig (string "active_suspension_mpc.png") :dpi 150)
     (print (string "Plot saved as active_suspension_mpc.png"))
     ;; (plt.show)
     )))
