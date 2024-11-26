import pennylane as qml
import torch.nn as nn

class BasicQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Define the quantum device
        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define the quantum circuit
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            # Encode the input data
            for i in range(self.n_qubits):
                qml.RY(inputs[:,i], wires=i)

            weights = weights.reshape(self.n_qubits, self.n_layers)
            
            for layer in range(self.n_layers):
                # Rotation layer
                for qubit in range(self.n_qubits):
                    qml.RX(weights[qubit][layer], wires=i)
                
                # Entanglement layer (CNOT ladder)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))
        # Initialize weights for the quantum circuit
        weight_shapes = {"weights": (self.n_qubits, self.n_layers)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        return self.qlayer(x)