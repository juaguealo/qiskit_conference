import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

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
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Definir el dispositivo cuántico
dev = qml.device("default.qubit", wires=4)

# 3. Definir el circuito cuántico para la QNN
@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    # Codificar las entradas clásicas como ángulos de rotación utilizando las tres rotaciones
    for i in range(4):
        qml.U3(inputs[i], inputs[i], inputs[i], wires=i)
    
    # Aplicar capas variacionales con rotaciones y puertas de entrelazamiento
    for layer_weights in weights:
        for i in range(4):
            qml.U3(layer_weights[i][0], layer_weights[i][1], layer_weights[i][2], wires=i)
        qml.broadcast(unitary=qml.CNOT, pattern='ring', wires=range(4))
    
    # Medir en la base Z
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# 4. Definir la capa cuántica personalizada
defining_layers = 5
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        # Inicializar los pesos cuánticos con más capas variacionales
        self.weights = nn.Parameter(torch.randn(defining_layers, 4, 3, requires_grad=True))

    def forward(self, x):
        # Aplicar el circuito cuántico a cada ejemplo del lote
        return torch.stack([torch.tensor(quantum_circuit(xi, self.weights), dtype=torch.float32) for xi in x])

# 5. Definir el modelo cuántico no híbrido
class QuantumOnlyQNN(nn.Module):
    def __init__(self):
        super(QuantumOnlyQNN, self).__init__()
        self.quantum_layer = QuantumLayer()

    def forward(self, x):
        # Obtener las expectativas directamente del circuito cuántico y aplicar una función sigmoide para clasificación binaria
        x = self.quantum_layer(x)
        x = torch.sigmoid(x.mean(dim=1, keepdim=True))
        return x

# 6. Instanciar el modelo, la función de pérdida y el optimizador
model = QuantumOnlyQNN()
criterion = nn.BCELoss()  # Clasificación binaria
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Ajuste de la tasa de aprendizaje

# 7. Entrenar la QNN no híbrida
n_epochs = 150  # Aumentar el número de épocas
losses = []
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# 8. Visualizar el circuito cuántico
print("\nCircuito cuántico:")
qml.draw_mpl(quantum_circuit)(X_train[0], model.quantum_layer.weights.detach().numpy())
plt.title("Circuito Cuántico de la QNN No Híbrida")
plt.show()

# 9. Visualizar la curva de aprendizaje
plt.plot(range(n_epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de Aprendizaje de la QNN No Híbrida')
plt.show()

# 10. Evaluar el modelo en el conjunto de prueba
model.eval()
y_pred_test = model(X_test).squeeze().detach().numpy().round()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clase 0', 'Clase 1'])
disp.plot()
plt.title('Matriz de Confusión de la QNN No Híbrida')
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