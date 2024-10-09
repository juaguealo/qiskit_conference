from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_circuit_layout
import matplotlib.pyplot as plt

# Crear un circuito cuántico simple
qc = QuantumCircuit(8)
for i in range(8):
    qc.h(i)  # Aplicar una puerta de Hadamard
    
qc.measure_all()  # Medir el qubit

# Dibujar el circuito
qc.draw(output='mpl')
plt.show()  # Mostrar el circuito

# Simulador cuántico AerSimulator
simulator = AerSimulator()

# Transpilar el circuito para el simulador
transpiled_circuit = transpile(qc, simulator)

# Ejecutar el circuito directamente sin ensamblar
result = simulator.run(transpiled_circuit, shots=1024).result()

# Mostrar los resultados
counts = result.get_counts()
print(f"Números aleatorios generados: {counts}")

# Mostrar el histograma
plot_histogram(counts, title='Resultados de estados cuánticos en binario')
plt.show()  # Mostrar el gráfico
