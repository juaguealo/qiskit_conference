from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, entropy

# Función para generar números aleatorios con el primer circuito (entrelazamiento y superposición)
def generate_numbers_entanglement():
    # Crear un circuito cuántico simple con 8 qubits
    qc = QuantumCircuit(8)

    # Aplicar la puerta de Hadamard a todos los qubits para generar superposición
    for i in range(8):
        qc.h(i)

    # Aplicar puertas CNOT para entrelazar los qubits
    for i in range(7):
        qc.cx(i, i + 1)

    # Medir todos los qubits
    qc.measure_all()

    # Simulador cuántico AerSimulator
    simulator = AerSimulator()

    # Transpilar el circuito para el simulador
    transpiled_circuit = transpile(qc, simulator)

    # Ejecutar el circuito directamente sin ensamblar
    result = simulator.run(transpiled_circuit, shots=1024).result()

    # Obtener los resultados y convertir a números decimales
    counts = result.get_counts()
    numbers = [int(binary_value, 2) for binary_value in counts.keys() for _ in range(counts[binary_value])]
    return numbers

# Función para generar números aleatorios con el segundo circuito (solo superposición)
def generate_numbers_superposition():
    # Crear un circuito cuántico simple con 8 qubits
    qc = QuantumCircuit(8)
    for i in range(8):
        qc.h(i)  # Aplicar una puerta de Hadamard
    
    # Medir todos los qubits
    qc.measure_all()

    # Simulador cuántico AerSimulator
    simulator = AerSimulator()

    # Transpilar el circuito para el simulador
    transpiled_circuit = transpile(qc, simulator)

    # Ejecutar el circuito
    result = simulator.run(transpiled_circuit, shots=1024).result()

    # Obtener los resultados y convertir a números decimales
    counts = result.get_counts()
    numbers = [int(binary_value, 2) for binary_value in counts.keys() for _ in range(counts[binary_value])]
    return numbers

# Generar números aleatorios de ambos métodos
numbers_entanglement = generate_numbers_entanglement()
numbers_superposition = generate_numbers_superposition()

# Prueba de Chi-cuadrado para comparar uniformidad
observed_entanglement, _ = np.histogram(numbers_entanglement, bins=range(2**8 + 1))
observed_superposition, _ = np.histogram(numbers_superposition, bins=range(2**8 + 1))

chi_stat_entanglement, p_val_entanglement = chisquare(observed_entanglement)
chi_stat_superposition, p_val_superposition = chisquare(observed_superposition)

print(f"Chi-cuadrado Método Entrelazamiento: estadístico={chi_stat_entanglement}, p-valor={p_val_entanglement}")
print(f"Chi-cuadrado Método Superposición: estadístico={chi_stat_superposition}, p-valor={p_val_superposition}")

# Entropía de Shannon
entropy_entanglement = entropy(observed_entanglement + 1e-10)  # Añadir una pequeña constante para evitar log(0)
entropy_superposition = entropy(observed_superposition + 1e-10)

print(f"Entropía Método Entrelazamiento: {entropy_entanglement}")
print(f"Entropía Método Superposición: {entropy_superposition}")

# Mostrar histogramas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(numbers_entanglement, bins=range(2**8 + 1), alpha=0.7, color='blue', label='Entrelazamiento')
plt.title('Distribución de Números - Entrelazamiento')
plt.xlabel('Número Decimal')
plt.ylabel('Frecuencia')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(numbers_superposition, bins=range(2**8 + 1), alpha=0.7, color='green', label='Superposición')
plt.title('Distribución de Números - Superposición')
plt.xlabel('Número Decimal')
plt.ylabel('Frecuencia')
plt.legend()

plt.tight_layout()
plt.show()