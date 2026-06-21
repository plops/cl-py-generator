#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <execution>
#include <thread>
#include "lambert_solver.h"

// Führt das Lösen des Lambert BVP und die Berechnung des Delta-V aus.
// Gibt true zurück, wenn das Newton-Verfahren konvergiert ist.
bool solve_lambert(double t_dep, double tof, double& out_dv, double& out_vx0, double& out_vy0) {
    double p[2] = { t_dep, tof };

    // Zustandsvariablen und Puffer
    double z[2] = { 0.0, 0.0 }; // Startgeschwindigkeit der Sonde
    double g[2] = { 0.0, 0.0 }; // Residuen (Endposition - Mars-Position)
    double J[4] = { 0.0, 0.0, 0.0, 0.0 }; // 2x2 Jacobimatrix
    double s_end[4] = { 0.0, 0.0, 0.0, 0.0 }; // Endzustand der Sonde
    double v_E[2] = { 0.0, 0.0 }; // Erdgeschwindigkeit
    double v_M[2] = { 0.0, 0.0 }; // Marsgeschwindigkeit

    const double* arg[2] = { z, p };
    double* res[5] = { g, J, s_end, v_E, v_M };

    // 1. Aufruf mit dummy z, um die aktuelle Erdgeschwindigkeit v_E zu ermitteln
    int status = lambert_eval(arg, res, nullptr, nullptr, 0);
    if (status != 0) return false;

    // Startschätzung setzen (Erdgeschwindigkeit + 10% Schub)
    z[0] = 1.1 * v_E[0];
    z[1] = 1.1 * v_E[1];

    // 2. Newton-Verfahren zur Lösung des Lambert-Problems
    bool converged = false;
    const int max_iters = 15;
    for (int iter = 0; iter < max_iters; ++iter) {
        status = lambert_eval(arg, res, nullptr, nullptr, 0);
        if (status != 0) return false;

        // Lineares 2x2 Gleichungssystem lösen: J * dz = g
        // J = [ J[0]  J[1] ]
        //     [ J[2]  J[3] ]
        double a = J[0], b = J[1], c = J[2], d = J[3];
        double det = a * d - b * c;
        if (std::abs(det) < 1e-12) return false;

        // Cramer'sche Regel für 2x2 Inverse
        double dz0 = (d * g[0] - b * g[1]) / det;
        double dz1 = (a * g[1] - c * g[0]) / det;

        z[0] -= dz0;
        z[1] -= dz1;

        // Konvergenzprüfung
        if (std::sqrt(dz0 * dz0 + dz1 * dz1) < 1e-6) {
            converged = true;
            break;
        }
    }

    if (converged) {
        // Evaluierung am Konvergenzpunkt
        status = lambert_eval(arg, res, nullptr, nullptr, 0);
        if (status != 0) return false;

        // Delta-V am Start und am Ziel berechnen
        double dv_dep = std::sqrt((z[0] - v_E[0]) * (z[0] - v_E[0]) + (z[1] - v_E[1]) * (z[1] - v_E[1]));
        double dv_arr = std::sqrt((s_end[2] - v_M[0]) * (s_end[2] - v_M[0]) + (s_end[3] - v_M[1]) * (s_end[3] - v_M[1]));

        // Konvertierungsfaktor von AU/yr in km/s
        const double V_CONV = 1.496e8 / (365.25 * 24 * 3600);
        out_dv = (dv_dep + dv_arr) * V_CONV;
        out_vx0 = z[0];
        out_vy0 = z[1];
        return true;
    }

    return false;
}

struct GridPoint {
    int i;
    int j;
};

int main() {
    std::cout << "========================================================================" << std::endl;
    std::cout << "Erde-Mars Porkchop-Plot C++ Simulator (STL Parallelized)" << std::endl;
    std::cout << "========================================================================" << std::endl;

    // Gittergröße festlegen (500x500 = 250.000 Punkte)
    const int N_dep = 500;
    const int N_tof = 500;

    std::vector<double> t_dep_grid(N_dep);
    for (int i = 0; i < N_dep; ++i) {
        t_dep_grid[i] = 0.0 + i * (2.5 - 0.0) / (N_dep - 1);
    }

    std::vector<double> tof_days_grid(N_tof);
    for (int j = 0; j < N_tof; ++j) {
        tof_days_grid[j] = 100.0 + j * (450.0 - 100.0) / (N_tof - 1);
    }

    // Gitterpunkte für STL parallel algorithm flach strukturieren
    std::vector<GridPoint> points;
    points.reserve(N_dep * N_tof);
    for (int i = 0; i < N_dep; ++i) {
        for (int j = 0; j < N_tof; ++j) {
            points.push_back({i, j});
        }
    }

    // Puffer für Ergebnisse
    std::vector<double> dv_results(N_dep * N_tof, 0.0);

    std::cout << "Starte Simulation auf " << std::thread::hardware_concurrency() 
              << " logischen Kernen..." << std::endl;
    std::cout << "Berechne " << N_dep * N_tof << " Trajektorien..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallelisierung über C++20 / STL Parallel Execution Policy (std::execution::par)
    std::for_each(std::execution::par, points.begin(), points.end(), [&](const GridPoint& pt) {
        int i = pt.i;
        int j = pt.j;
        double t_dep = t_dep_grid[i];
        double tof = tof_days_grid[j] / 365.25;

        double dv = 0.0, vx0 = 0.0, vy0 = 0.0;
        bool success = solve_lambert(t_dep, tof, dv, vx0, vy0);

        if (success) {
            dv_results[i * N_tof + j] = dv;
        } else {
            dv_results[i * N_tof + j] = NAN;
        }
    });

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "Simulation erfolgreich abgeschlossen in " << duration.count() 
              << " ms." << std::endl;

    // Speichern der Ergebnisse in CSV
    std::string csv_filename = "porkchop_cpp.csv";
    std::ofstream csv_file(csv_filename);
    if (csv_file.is_open()) {
        csv_file << "t_dep,tof_days,dv_tot\n";
        for (int i = 0; i < N_dep; ++i) {
            for (int j = 0; j < N_tof; ++j) {
                double dv = dv_results[i * N_tof + j];
                if (!std::isnan(dv) && dv < 40.0) { // Plausibilitätsfilter
                    csv_file << t_dep_grid[i] << "," 
                             << tof_days_grid[j] << "," 
                             << dv << "\n";
                }
            }
        }
        csv_file.close();
        std::cout << "Ergebnisse gespeichert in: " << csv_filename << std::endl;
    } else {
        std::cerr << "Fehler beim Öffnen der CSV-Datei!" << std::endl;
    }

    return 0;
}
