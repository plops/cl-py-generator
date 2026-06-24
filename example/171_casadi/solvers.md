CasADi bietet Schnittstellen zu zahlreichen NLP-, QP- und SDP-Solvern an, darunter kommerzielle und Open-Source-Alternativen. Für die Kombination aus Moving Horizon Estimation (MHE) und Model Predictive Control (MPC) bei einem invertierten Pendel ist hpipm (über eine QP-Formulierung) oder acados die performanteste Wahl, während osqp (für lineare MPC/MHE) oder ein hochgradig optimierter ipopt die besten Standardoptionen darstellen. [1] 
Hier ist die strukturierte Übersicht aller verfügbaren Solver-Schnittstellen in CasADi und die optimalen Empfehlungen für Ihr Pendel-Projekt. [2] 
------------------------------
## 🌐 Übersicht der CasADi-Solver nach Problemtyp
CasADi löst Optimierungsprobleme, indem es sie intern in standardisierte mathematische Strukturen gießt. Je nach Problemtyp stehen Ihnen folgende Schnittstellen zur Verfügung: [3, 4] 
## Non-Linear Programming (NLP) Solver
Für nichtlineare Systemdynamiken (z. B. große Auslenkungen des Pendels, Sinus-/Cosinus-Terme):

* ipopt: Der extrem robuste Open-Source-Standard (Innere-Punkte-Methode).
* snopt: Kommerzieller SQP-Solver (Sequential Quadratic Programming), sehr effizient bei teuren Funktionsauswertungen.
* worhp: Kommerzieller Solver für dünnbesetzte, sehr große nichtlineare Probleme.
* blocksqp: Ein modularer SQP-Solver für strukturierte Probleme. [5, 6, 7] 

## Quadratic Programming (QP) Solver [8] 
Falls Sie Ihr Pendel linearisieren (Linear MPC / Linear MHE) oder ein SQP-Verfahren nutzen: [9, 10] 

* hpipm: Extrem schneller, auf eingebettete Systeme optimierter QCQP/QP-Solver (nutzt BLASFEO).
* osqp: Moderner, sehr schneller Open-Source-Solver basierend auf ADMM, ideal für Echtzeit-MPC.
* qpoases: Verlässlicher Online-Aktive-Mengen-Solver, perfekt für kleine bis mittlere MPC-Probleme mit schnellem Warmstart.
* gurobi / cplex: High-End-Kommerzsolver, extrem schnell und stabil.
* clp / ooqp: Solide Open-Source-Alternativen für lineare/quadratische Programme. [11, 12] 

------------------------------
## 🏆 Die besten Solver für MHE + MPC eines invertierten Pendels
Da ein invertiertes Pendel eine schnelle Dynamik aufweist (Abtastraten von oft 10 ms bis 50 ms erforderlich), ist die Rechenzeit kritisch. Die Wahl hängt von Ihrer Systemformulierung ab:

| Rang [13, 14, 15] | Solver | Problemtyp | Warum für invertiertes Pendel? |
|---|---|---|---|
| 1 | hpipm / acados | Strukturierte NLP/QP | Maximale Geschwindigkeit. hpipm nutzt die Baumstruktur von MHE/MPC perfekt aus. Falls möglich, nutzen Sie direkt das mit CasADi verwandte Framework acados[](https://docs.acados.org/), da es Code-Generierung für Pendel-Echtzeitregelungen im Mikrosekundenbereich erlaubt. |
| 2 | osqp | Linear QP | Beste Open-Source-Wahl für lineare MPC. Wenn Sie das Pendel um die obere Ruhelage linearisieren, löst osqp das MPC- und MHE-Problem extrem zuverlässig und in wenigen Millisekunden. |
| 3 | ipopt | Allgemeines NLP | Beste Wahl für voll-nichtlineare Dynamik (Standard). ipopt fängt auch extreme Stürze des Pendels ab, wo Linearisierungen scheitern. Für Echtzeitanwendungen müssen Sie jedoch die maximale Iterationszahl strikt limitieren. |
| 4 | qpoases | Kleines QP | Perfekt für klassisches MPC. Es nutzt "Warmstarts" (Lösung des vorherigen Zeitschritts als Startwert). Da sich das Pendel kontinuierlich bewegt, findet qpoases die neue Lösung oft in unter 5 Iterationen. |

------------------------------
## 🛠️ Code-Beispiele für die Praxis
So konfigurieren Sie die empfohlenen Solver in Ihrer CasADi-Syntax:
## Option A: Nichtlineares MPC/MHE mit optimiertem ipopt (Echtzeit-Tricks)
Wenn Sie bei ipopt bleiben, müssen Sie die Iterationen und Toleranzen begrenzen, damit der Regler die Abtastzeit des Pendels nicht reißt:

opts_nlp = {
    "print_time": False,
    "ipopt.print_level": 0,
    "ipopt.max_iter": 20,          # Verhindert endlose Schleifen in Echtzeit
    "ipopt.tol": 1e-4,             # Leicht gelockerte Toleranz für mehr Speed
    "ipopt.acceptable_tol": 1e-3,
    "ipopt.hessian_approximation": "limited-memory" # Spart Rechenzeit für exakte Hessematrizen
}
opti_mhe.solver("ipopt", opts_nlp)

## Option B: Lineares MPC/MHE mit osqp (Sehr schnell, falls linearisiert)

# OSQP erwartet eine rein quadratische/lineare Formulierung (Opti QP)opts_qp = {
    "print_time": False,
    "osqp.verbose": False,
    "osqp.max_iter": 4000,
    "osqp.eps_abs": 1e-4,
    "osqp.eps_rel": 1e-4
}
opti_mhe.solver("osqp", opts_qp)

## Option C: Die embedded Profilösung qpoases

opts_qpoases = {
    "print_time": False,
    "printLevel": "none"
}
opti_mhe.solver("qpoases", opts_qpoases)

------------------------------
Wenn Sie maximale Performance auf echter Hardware benötigen, empfiehlt es sich, die Systemgleichungen in CasADi zu definieren, diese aber per C-Code-Generierung in acados zu exportieren, welches explizit für embedded MPC/MHE auf Systemen wie dem invertierten Pendel gebaut wurde.
Möchten Sie das Pendel vollständig nichtlinear regeln (z.B. für einen Swing-Up aus dem freien Hängen) oder regeln Sie es nur in der oberen Ruhelage (wo eine Linearisierung ausreicht)? [16] 

[1] [https://web.casadi.org](https://web.casadi.org/docs/)
[2] [https://www.syscop.de](https://www.syscop.de/files/2023ss/MPC4RES/MPCforRES_Exercise2_sol.pdf)
[3] [https://web.casadi.org](https://web.casadi.org/docs/)
[4] [https://medium.com](https://medium.com/@shoaib6174/an-introduction-to-casadi-with-python-12055f8e652f)
[5] [https://www.syscop.de](https://www.syscop.de/files/2025ss/rmpc25/exercises/exercise2.pdf)
[6] [https://adamseewald.cc](https://adamseewald.cc/teaching/optimization-control/nlp-solvers/)
[7] [https://www.researchgate.net](https://www.researchgate.net/publication/261081671_Dynamic_optimization_with_CasADi)
[8] [https://web.casadi.org](https://web.casadi.org/docs/)
[9] [https://web.casadi.org](https://web.casadi.org/docs/)
[10] [https://lirias.kuleuven.be](https://lirias.kuleuven.be/retrieve/b19a85ed-8044-40f1-bff0-b7164c0353f5)
[11] [https://discourse.acados.org](https://discourse.acados.org/t/solving-simple-nlp-problem-exploit-blasfeo-performance/271)
[12] [https://web.casadi.org](https://web.casadi.org/docs/)
[13] [https://www.syscop.de](https://www.syscop.de/files/2025ss/rmpc25/exercises/exercise2.pdf)
[14] [https://www.syscop.de](https://www.syscop.de/files/2023ss/MPC4RES/MPCforRES_Exercise2_sol.pdf)
[15] [https://www.syscop.de](https://www.syscop.de/files/2025ss/rmpc25/exercises/exercise2.pdf)
[16] [https://www.mdpi.com](https://www.mdpi.com/1996-1073/16/5/2143)
