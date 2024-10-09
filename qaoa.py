import networkx as nx
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Crear un grafo simple para resolver el problema de Max-Cut
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

# Mostrar el grafo
nx.draw(G, with_labels=True)
plt.show()

# Definir el problema Max-Cut en Qiskit
max_cut = Maxcut(G)
qubo = max_cut.to_quadratic_program()

# Convertir el problema a QUBO (formato para optimización cuántica)
converter = QuadraticProgramToQubo()
qubo_problem = converter.convert(qubo)

# Configurar el simulador cuántico
simulator = AerSimulator()

# Configurar el optimizador clásico y el QAOA utilizando el Sampler
optimizer = COBYLA()
sampler = Sampler()
qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)

# Resolver el problema usando el optimizador cuántico
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
result = qaoa_optimizer.solve(qubo_problem)

# Mostrar el resultado
print(f"Solución óptima: {result.x}")
print(f"Valor óptimo: {result.fval}")
