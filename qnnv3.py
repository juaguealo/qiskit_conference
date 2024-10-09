import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.optim import Adam
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

# Configurar el backend de matplotlib
plt.switch_backend('Agg')

# ---------------------------------
# 1. Cargar y preparar el Iris Dataset
# ---------------------------------
data = load_iris()
X = data.data
y = data.target

# Normalizar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convertir las etiquetas en formato de tensor de PyTorch
y_scaled = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convertir los conjuntos de entrenamiento a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# ---------------------------------
# 2. Crear el circuito cuántico parametrizado
# ---------------------------------
n_qubits = 4  # Cambiamos el número de qubits a 4 (Iris Dataset tiene 4 características)
# Crear dos conjuntos de parámetros: uno para las entradas y otro para los pesos
input_params = ParameterVector('input', n_qubits)  # Parámetros de entrada con 4 parámetros
weight_params = ParameterVector('weight', 16)  # Aumentar los parámetros de pesos a 16

# Crear un circuito cuántico más profundo
qc = QuantumCircuit(n_qubits)
for i in range(n_qubits):
    qc.rx(input_params[i], i)
    qc.ry(weight_params[2 * i], i)
    if i < n_qubits - 1:
        qc.cx(i, i + 1)
    qc.rz(weight_params[2 * i + 1], i)

# Añadir otra capa de rotaciones y entrelazamiento
for i in range(n_qubits):
    qc.ry(weight_params[2 * i + 8], i)
    qc.rz(weight_params[2 * i + 9], i)
    if i < n_qubits - 1:
        qc.cx(i, i + 1)

# Definir el observable: mediremos en la base Z para todos los qubits
observable = SparsePauliOp.from_list([("Z" * n_qubits, 1)])

# ---------------------------------
# 3. Crear la Red Neuronal Cuántica con EstimatorQNN
# ---------------------------------
# Usamos EstimatorQNN con el circuito cuántico parametrizado y el observable
simulator = AerSimulator()
qnn = EstimatorQNN(circuit=qc, observables=observable, input_params=input_params, weight_params=weight_params)

# Conectar la QNN a PyTorch
model = TorchConnector(qnn)

# Definir el modelo de PyTorch
class QuantumNet(nn.Module):
    def __init__(self):
        super(QuantumNet, self).__init__()
        self.qnn_layer = model

    def forward(self, x):
        return self.qnn_layer(x)

# Crear el modelo cuántico
quantum_model = QuantumNet()

# ---------------------------------
# 4. Entrenar el modelo cuántico
# ---------------------------------
# Definir el optimizador y la función de pérdida
optimizer = Adam(quantum_model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()  # Cambiamos a CrossEntropy para clasificación multiclase

# Definir número de épocas
epochs = 20
loss_history = []

# Entrenar el modelo
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = quantum_model(X_train)
    loss = loss_function(y_pred, y_train.squeeze().long())  # Convertimos a enteros
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    print(f"Época {epoch+1}/{epochs}, Pérdida: {loss.item()}")

# ---------------------------------
# 5. Guardar la gráfica de la evolución de la pérdida
# ---------------------------------
plt.plot(loss_history)
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.savefig('loss_history.png')  # Guardar la gráfica como imagen

# ---------------------------------
# 6. Evaluar el modelo en el conjunto de prueba
# ---------------------------------
quantum_model.eval()
y_test_pred = quantum_model(X_test)

# Convertir las predicciones usando argmax para multiclase
y_test_pred = torch.argmax(y_test_pred, axis=1)  # Usamos argmax para obtener la clase más probable
y_test_pred = y_test_pred.detach().numpy()

# Convertir las etiquetas a enteros
y_test_labels = y_test.squeeze().long().numpy()

# ---------------------------------
# 7. Calcular la precisión y mostrar métricas
# ---------------------------------
accuracy = accuracy_score(y_test_labels, y_test_pred)
conf_matrix = confusion_matrix(y_test_labels, y_test_pred)
class_report = classification_report(y_test_labels, y_test_pred)

print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Matriz de confusión:")
print(conf_matrix)
print("Reporte de clasificación:")
print(class_report)
