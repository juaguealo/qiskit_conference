import networkx as nx
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.quantum_info import SparsePauliOp

# Crear un grafo más complejo con 20 nodos y 40 aristas aleatorias
G = nx.gnm_random_graph(20, 40)

# Mostrar el grafo
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color="lightblue", node_size=700, font_size=12)
plt.title("Grafo para el problema de Max-Cut con 20 nodos")
plt.show()

# Definir el problema Max-Cut en Qiskit
max_cut = Maxcut(G)
qubo = max_cut.to_quadratic_program()

# Convertir el problema a QUBO (formato para optimización cuántica)
converter = QuadraticProgramToQubo()
qubo_problem = converter.convert(qubo)

# Configurar el simulador cuántico
simulator = AerSimulator()

# Definir un observable válido usando SparsePauliOp
pauli_observable = SparsePauliOp.from_list([("ZZ", 1)])

# Utilizar Estimator con el observable correcto
optimizer = COBYLA()
estimator = Sampler()
qaoa = QAOA(sampler=estimator, optimizer=optimizer, reps=1)

# Resolver el problema usando el optimizador cuántico
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
result = qaoa_optimizer.solve(qubo_problem)

# Mostrar el resultado
print(f"Solución óptima: {result.x}")
print(f"Valor óptimo: {result.fval}")

# Obtener la solución de Max-Cut
cut = max_cut.interpret(result)

# Inicializar los colores de los nodos en 'green' por defecto
node_colors = ['green'] * len(G.nodes)

# Asignar 'red' a los nodos en la segunda partición
for partition in cut:
    for node in partition:
        node_colors[node] = 'red' if partition == cut[1] else 'green'

# Graficar el resultado
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color=node_colors, node_size=700, font_size=12)
plt.title("Solución de Max-Cut con 20 nodos")
plt.show()
