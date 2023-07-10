import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib

# Define geometry
nodes = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
elements = np.array([[0, 1], [1, 2], [2, 0]])

# Material properties
E = 210000  # Young's modulus in N/mm^2
A = 1  # cross-sectional area in mm^2

# Define global stiffness matrix
K = lil_matrix((2*len(nodes), 2*len(nodes)))

# Assembly of global stiffness matrix
for i, j in elements:
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
    c = (xj-xi) / L
    s = (yj-yi) / L
    k_local = E*A/L * np.array([[c**2, c*s, -c**2, -c*s],
                                [c*s, s**2, -c*s, -s**2],
                                [-c**2, -c*s, c**2, c*s],
                                [-c*s, -s**2, c*s, s**2]])
    K[2*i:2*i+2, 2*i:2*i+2] += k_local[:2, :2]
    K[2*i:2*i+2, 2*j:2*j+2] += k_local[:2, 2:]
    K[2*j:2*j+2, 2*i:2*i+2] += k_local[2:, :2]
    K[2*j:2*j+2, 2*j:2*j+2] += k_local[2:, 2:]

# Convert to CSC format for efficient solving
K = csc_matrix(K)

# Force vector
F = np.zeros(2*len(nodes))
F[4] = -1000  # force at the top node in y direction

# Boundary conditions
bc = np.zeros(2*len(nodes), dtype=bool)
bc[0] = bc[1] = bc[2] = True  # fixed at the left end
bc[3] = True  # free in y direction at the right end

# Apply boundary conditions
K[:, bc] = K[bc, :] = 0
K[bc, bc] = 1
F[bc] = 0

# Solve Ku = F
u = spsolve(K, F)

# Print displacements
for i in range(len(nodes)):
    print(f"Node {i}: displacement = {u[2*i:2*i+2]}")

# Plot deformed shape
#deformed_nodes = nodes + u.reshape(-1, 2)
#plt.triplot(nodes[:, 0], nodes[:, 1], elements, label='Original')
#plt.triplot(deformed_nodes[:, 0], deformed_nodes[:, 1], elements, label='Deformed')
#plt.legend()
#plt.show()


# Plot deformed shape
deformed_nodes = nodes + u.reshape(-1, 2)

for element in elements:
    # Original shape
    plt.plot(*nodes[element].T, 'b-', label='Original')
    # Deformed shape
    plt.plot(*deformed_nodes[element].T, 'r-', label='Deformed')

# Because each line gets its own label, we'll fix that here
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
