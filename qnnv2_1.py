import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.optim import Adam
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit.quantum_info import SparsePauliOp

# ---------------------------------
# 1. Cargar y preparar el Iris Dataset
# ---------------------------------
data = load_iris()
X = data.data
y = data.target

# Aplicar StandardScaler para estandarizar los datos
scaler = StandardScaler()
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
n_qubits = 4
input_params = ParameterVector('input', n_qubits)
weight_params = ParameterVector('weight', 8)

# Crear un circuito cuántico
qc = QuantumCircuit(n_qubits)
for i in range(n_qubits):
    qc.rx(input_params[i], i)
    qc.ry(weight_params[2 * i], i)
    if i < n_qubits - 1:
        qc.cx(i, i + 1)
    qc.rz(weight_params[2 * i + 1], i)

# Definir el observable
observable = SparsePauliOp.from_list([("Z" * n_qubits, 1)])

# ---------------------------------
# 3. Crear la Red Neuronal Cuántica con EstimatorQNN
# ---------------------------------
simulator = AerSimulator()
qnn = EstimatorQNN(circuit=qc, observables=observable, input_params=input_params, weight_params=weight_params)
model = TorchConnector(qnn)

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
optimizer = Adam(quantum_model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

# Definir número de épocas
epochs = 20
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = quantum_model(X_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    print(f"Época {epoch+1}/{epochs}, Pérdida: {loss.item()}")

# ---------------------------------
# 5. Guardar la curva de aprendizaje
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Pérdida')
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig('loss_history.png')  # Guardar la gráfica como imagen

# ---------------------------------
# 6. Evaluar el modelo en el conjunto de prueba
# ---------------------------------
quantum_model.eval()
y_test_pred = quantum_model(X_test)

y_test_pred = torch.round(y_test_pred)  # Aproximamos la salida a 0 o 1
y_test_pred = y_test_pred.detach().numpy()

y_test_labels = y_test.detach().numpy().astype(int)
y_pred_labels = y_test_pred.astype(int)

# ---------------------------------
# 7. Calcular la precisión y mostrar métricas
# ---------------------------------
accuracy = accuracy_score(y_test_labels, y_pred_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
class_report = classification_report(y_test_labels, y_pred_labels, zero_division=1)

print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Matriz de confusión:")
print(conf_matrix)
print("Reporte de clasificación:")
print(class_report)

# ---------------------------------
# 8. Graficar la matriz de confusión
# ---------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.savefig('confusion_matrix.png')  # Guardar la matriz de confusión como imagen
