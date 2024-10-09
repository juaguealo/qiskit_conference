import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Function
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler
# Importar clases necesarias para manejo cuántico
# from qiskit.quantum_info import PauliZ (removido porque no se encuentra en esta versión)
from qiskit.circuit import Parameter
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA


# 1. Cargar y preparar el conjunto de datos Iris
data = load_iris()
X = data.data  # Características
y = data.target  # Etiquetas

# Filtramos para una clasificación binaria entre clases 0 y 1 (para simplificar el problema)
X = X[y != 2]
y = y[y != 2]

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1)

# 2. Definir el dispositivo cuántico
backend = AerSimulator()
sampler = StatevectorSampler()

# 3. Definir el circuito cuántico para la QNN
def create_quantum_circuit(inputs, weights):
    num_qubits = len(inputs)
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Codificar las entradas como ángulos de rotación
    for i in range(num_qubits):
        qc.ry(inputs[i], i)
    
    # Aplicar capas variacionales
    weight_idx = 0
    for i in range(num_qubits):
        qc.ry(weights[weight_idx], i)
        weight_idx += 1
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    qc.cz(num_qubits - 1, 0)

    # Agregar medida
    for i in range(num_qubits):
        qc.measure(i, i)

    return qc

# 4. Definir la capa cuántica personalizada utilizando torch.autograd.Function
class QuantumCircuitFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        inputs_np = inputs.detach().numpy()
        weights_np = weights.detach().numpy()
        circuit = create_quantum_circuit(inputs_np, weights_np)
        result = backend.run(transpile(circuit, backend)).result()
        counts = result.get_counts()
        expectation = (sum((-1 if key.count('1') % 2 else 1) * count for key, count in counts.items()) / 1024 + 1) / 2  # Normalizar a [0, 1]
        ctx.save_for_backward(inputs, weights)
        return torch.tensor(expectation, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        grad_inputs = grad_weights = None

        # Aproximación de gradiente numérico para inputs
        if ctx.needs_input_grad[0]:
            grad_inputs = torch.zeros_like(inputs)
            epsilon = 1e-3
            for i in range(len(inputs)):
                inputs_eps = inputs.clone()
                inputs_eps[i] += epsilon
                expectation_pos = QuantumCircuitFunction.apply(inputs_eps, weights)

                inputs_eps[i] -= 2 * epsilon
                expectation_neg = QuantumCircuitFunction.apply(inputs_eps, weights)

                grad_inputs[i] = (expectation_pos - expectation_neg) / (2 * epsilon)

        # Aproximación de gradiente numérico para weights
        if ctx.needs_input_grad[1]:
            grad_weights = torch.zeros_like(weights)
            epsilon = 1e-3
            for i in range(len(weights)):
                weights_eps = weights.clone()
                weights_eps[i] += epsilon
                expectation_pos = QuantumCircuitFunction.apply(inputs, weights_eps)

                weights_eps[i] -= 2 * epsilon
                expectation_neg = QuantumCircuitFunction.apply(inputs, weights_eps)

                grad_weights[i] = (expectation_pos - expectation_neg) / (2 * epsilon)

        grad_inputs = grad_inputs * grad_output if grad_inputs is not None else None
        grad_weights = grad_weights * grad_output if grad_weights is not None else None
        return grad_inputs, grad_weights

# 5. Definir el modelo cuántico no híbrido
class QuantumOnlyQNN(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumOnlyQNN, self).__init__()
        self.num_qubits = num_qubits
        self.weights = nn.Parameter(torch.randn(num_qubits, requires_grad=True))

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            inputs = x[i]
            output = QuantumCircuitFunction.apply(inputs, self.weights)
            outputs.append(output)
        return torch.sigmoid(torch.stack(outputs))

# 6. Instanciar el modelo, la función de pérdida y el optimizador
num_qubits = 4
model = QuantumOnlyQNN(num_qubits)
criterion = nn.BCELoss()  # Clasificación binaria
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Entrenar la QNN no híbrida
n_epochs = 50  # Reducir el número de épocas para pruebas iniciales
losses = []
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")

# 8. Visualizar la curva de aprendizaje
plt.plot(range(n_epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de Aprendizaje de la QNN No Híbrida con Qiskit')
plt.show()

# 9. Evaluar el modelo en el conjunto de prueba
model.eval()
y_pred_test = model(X_test).detach().numpy().round()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clase 0', 'Clase 1'])
disp.plot()
plt.title('Matriz de Confusión de la QNN No Híbrida con Qiskit')
plt.show()

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"Exactitud: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Sensibilidad (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")