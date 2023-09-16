#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const int N = 1024;
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    // Initialize
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Offloading to GPU
    #pragma omp target map(to: a[0:N], b[0:N]) map(from: c[0:N])
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }

    // Verification
    for (int i = 0; i < N; ++i) {
        if (std::fabs(c[i] - 3.0f) > 1e-6) {
            std::cout << "Verification failed at index " << i << std::endl;
            delete[] a;
            delete[] b;
            delete[] c;
            return 1;
        }
    }

    std::cout << "Verification passed" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
