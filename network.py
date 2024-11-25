import pennylane as qml
import torch.nn as nn

class BasicQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = 2
        # Define the quantum device
        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define the quantum circuit
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Encode the input data
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Alternating layers of entanglement and rotation
            n_layers = 2
            weights = weights.reshape(n_layers, n_qubits, 3)
            
            for layer in range(n_layers):
                # Rotation layer
                for i in range(n_qubits):
                    qml.RX(weights[layer][i][0], wires=i)
                    qml.RY(weights[layer][i][1], wires=i)
                    qml.RZ(weights[layer][i][2], wires=i)
                
                # Entanglement layer (CNOT ladder)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            
            return qml.expval(qml.PauliZ(0))
        # Initialize weights for the quantum circuit
        weight_shapes = {"weights": (n_layers * n_qubits * 3,)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        return self.qlayer(x)