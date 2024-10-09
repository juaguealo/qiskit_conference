from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_circuit_layout
import matplotlib.pyplot as plt

# Crear un circuito cuántico simple con 8 qubits
qc = QuantumCircuit(8)
for i in range(0, 8):
    qc.h(i)  # Aplicar una puerta de Hadamard
    
qc.measure_all()  # Medir todos los qubits

# Dibujar el circuito
qc.draw(output='mpl')
plt.show()  # Mostrar el circuito

# Simulador cuántico AerSimulator
simulator = AerSimulator()

# Transpilar el circuito para el simulador
transpiled_circuit = transpile(qc, simulator)

# Ejecutar el circuito
result = simulator.run(transpiled_circuit, shots=1024).result()

# Obtener los resultados y convertir a números decimales
counts = result.get_counts()

# Mostrar los resultados en binario y decimal
for binary_value, count in counts.items():
    decimal_value = int(binary_value, 2)
    print(f"Binario: {binary_value}, Decimal: {decimal_value}, Repeticiones: {count}")

# Mostrar el histograma
plot_histogram(counts, title='Resultados de estados cuánticos en binario')
plt.show()  # Mostrar el gráfico
