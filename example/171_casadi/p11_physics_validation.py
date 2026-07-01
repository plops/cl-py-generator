import numpy as np

# Physical parameters
M = 1.0     # Cart mass
m = 0.1     # Pendulum mass
l = 0.5     # Pendulum length
g = 9.81    # Gravity

def get_eigenvalues(gravity_sign):
    # Linearized dynamics dx/dt = A x
    # x = [s, v, theta, omega]
    # For F = 0, at theta = 0 (upright):
    # den = M
    # dv/dt = (m * g * gravity_sign * theta) / M
    # domega/dt = ((M + m) * g * gravity_sign * theta) / (l * M)
    #
    # Let's write the A matrix for theta = 0:
    # A = [ 0  1  0  0 ]
    #     [ 0  0  a21 0 ]
    #     [ 0  0  0  1 ]
    #     [ 0  0  a41 0 ]
    
    if gravity_sign == 'current':
        a21 = m * g / M
        a43 = -(M + m) * g / (l * M)
    else:
        a21 = -m * g / M
        a43 = (M + m) * g / (l * M)
        
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, a21, 0],
        [0, 0, 0, 1],
        [0, 0, a43, 0]
    ])
    
    eigenvals = np.linalg.eigvals(A)
    return eigenvals

print("--- Eigenvalues at Theta = 0 (Upright) ---")
print("Current Code:  ", get_eigenvalues('current'))
print("Correct Physics:", get_eigenvalues('corrected'))

def get_eigenvalues_pi(gravity_sign):
    if gravity_sign == 'current':
        a21 = m * g / M
        a43 = (M + m) * g / (l * M)
    else:
        a21 = -m * g / M
        a43 = -(M + m) * g / (l * M)
        
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, a21, 0],
        [0, 0, 0, 1],
        [0, 0, a43, 0]
    ])
    
    eigenvals = np.linalg.eigvals(A)
    return eigenvals

print("\n--- Eigenvalues at Theta = pi (Hanging Down) ---")
print("Current Code:  ", get_eigenvalues_pi('current'))
print("Correct Physics:", get_eigenvalues_pi('corrected'))
