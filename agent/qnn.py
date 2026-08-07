from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, XGate, RXGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer.primitives import Sampler
from qiskit_aer import AerSimulator

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

from agent.encoding import AngleStateEncoder, BasisEncoder, AmplitudeEncoder


def crx(param_name: str = 'crx_gate') -> Gate:
    """
    Returns a controlled-RX gate with a single control qubit.
    """
    param = Parameter(param_name)
    return RXGate(param).control(1)


def get_twolocal_circuit(n_qubits, n_reps):
    """
    Constructs a two-local ansatz circuit without measurements.
    """
    qc = QuantumCircuit(n_qubits)
    qc.compose(RealAmplitudes(n_qubits, reps=n_reps), inplace=True)
    return qc


def get_bqn_circuit(n_qubits: int, n_data_qubits: int, n_ancilla_qubits: int,
                    n_data_reps: int, n_ancilla_reps: int) -> QuantumCircuit:
    """
    Builds a Bayesian Quantum Circuit with ancilla qubits and a flag qubit.
    """
    assert n_qubits == (n_data_qubits + n_ancilla_qubits + 1), "Total qubits must equal data plus ancilla plus one flag qubit"

    data_reg = QuantumRegister(n_data_qubits, 'data')
    ancilla_reg = QuantumRegister(n_ancilla_qubits + 1, 'ancilla')
    output_reg = ClassicalRegister(n_data_qubits, 'output')

    qc = QuantumCircuit(data_reg, ancilla_reg, output_reg)
    ancilla = get_bqn_ancilla(n_ancilla_qubits, n_ancilla_reps)
    ANCILLA_QUBITS = list(range(n_data_qubits, n_data_qubits + n_ancilla_qubits))
    qc.compose(ancilla, ANCILLA_QUBITS, inplace=True)

    for i in range(n_data_reps):
        circuit = get_one_bqn_repetition(n_data_qubits, n_ancilla_qubits, i)
        qc.compose(circuit, inplace=True)

    qc.barrier()
    return qc


def get_one_bqn_repetition(n_data_qubits: int, n_ancilla_qubits: int, block: int) -> QuantumCircuit:
    """
    Creates a single repetition of blocks controlled by the ancilla state.
    """
    FMT_STRING = f'0{n_ancilla_qubits}b'
    ANCILLA_QUBITS = list(range(n_data_qubits, n_data_qubits + n_ancilla_qubits + 1))

    qc = QuantumCircuit(n_data_qubits + n_ancilla_qubits + 1)

    for i in range(2**n_ancilla_qubits):
        selector, inverter = get_ancilla_selector(format(i, FMT_STRING), n_ancilla_qubits)
        block_circuit = get_one_bqn_block(n_data_qubits, n_ancilla_qubits, block, i)

        qc.compose(selector, ANCILLA_QUBITS, inplace=True)
        qc.compose(block_circuit, inplace=True)
        qc.compose(inverter, ANCILLA_QUBITS, inplace=True)

    return qc


def get_one_bqn_block(n_data_qubits: int, n_ancilla_qubits: int, block: int, rep: int) -> QuantumCircuit:
    """
    Constructs a single block for the Bayesian Quantum Circuit.
    """
    FLAG_BIT = n_data_qubits + n_ancilla_qubits

    qc = QuantumCircuit(n_data_qubits + n_ancilla_qubits + 1)

    for i in range(n_data_qubits):
        qc.append(crx(f'data_{block}_{rep}_{i}'), [FLAG_BIT, i])
    for i in range(n_data_qubits):
        qc.ccx(i, FLAG_BIT, (i + 1) % n_data_qubits)

    return qc


def get_bqn_ancilla(n_ancilla_qubits: int, n_ancilla_reps: int) -> QuantumCircuit:
    """
    Builds the ancilla register circuit for the Bayesian Quantum Circuit.
    """
    qc = QuantumCircuit(n_ancilla_qubits)

    for i in range(n_ancilla_reps):
        for j in range(n_ancilla_qubits):
            weight = Parameter(f'ancilla_{i}_{j}')
            qc.rx(weight, j)
        if n_ancilla_qubits > 1:
            for j in range(n_ancilla_qubits):
                qc.cx(j, (j + 1) % n_ancilla_qubits)

    return qc


def get_ancilla_selector(bitstring: str, n_ancilla_qubits: int):
    """
    Creates selection and inversion circuits to conditionally apply blocks.
    """
    qr = QuantumRegister(n_ancilla_qubits + 1)
    bitstring_circuit = QuantumCircuit(qr)
    selection_circuit = QuantumCircuit(qr)
    inversion_circuit = QuantumCircuit(qr)
    bits = [True if b == '1' else False for b in bitstring]

    for idx, bit in enumerate(bits):
        if bit:
            bitstring_circuit.x(idx)

    flag_set = XGate().control(n_ancilla_qubits)

    selection_circuit.compose(bitstring_circuit, inplace=True)
    selection_circuit.append(flag_set, qr)

    inversion_circuit.append(flag_set, qr)
    inversion_circuit.compose(bitstring_circuit, inplace=True)

    return selection_circuit, inversion_circuit


class TruncateOutputLayer(nn.Module):
    """
    Output layer that truncates the network output to the required elements.
    """
    def __init__(self, n_actions: int):
        super(TruncateOutputLayer, self).__init__()
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the required columns of the input tensor."""
        return torch.narrow(x, 1, 0, self.n_actions)

class QuantumDQN(nn.Module):
    """
    Quantum DQN using a parameterized quantum circuit.
    This implementation uses the statevector simulator directly via AerSimulator,
    avoiding the Sampler primitive and its transpilation issues.
    """
    def __init__(self, n_inputs, n_qubits, n_actions, param_layers=3,
                 qnn_type='twolocal', n_ancilla_bits=-1, n_ancilla_reps=-1,
                 encoding='angle', qnn_output='trunc', n_shots=1024,
                 torch_device='cpu'):
        super(QuantumDQN, self).__init__()

        # Validate arguments
        assert qnn_type in ['twolocal', 'bayes'], "Quantum network type must be twolocal or bayes"
        assert encoding in ['angle', 'basis', 'amplitude'], "Unknown encoding type"
        assert qnn_output in ['trunc', 'layer'], "Output type must be trunc or layer"

        n_data_qubits = n_qubits if qnn_type == 'twolocal' else n_qubits - n_ancilla_bits - 1

        # State encoder
        if encoding == 'angle':
            self.encoder = AngleStateEncoder(n_inputs, n_data_qubits, torch_device)
        elif encoding == 'basis':
            self.encoder = BasisEncoder()
        elif encoding == 'amplitude':
            self.encoder = AmplitudeEncoder()
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        # Quantum circuit
        from qiskit.circuit.library import ZZFeatureMap

        self.feature_map = ZZFeatureMap(n_data_qubits)
        if qnn_type == 'twolocal':
            self.ansatz = get_twolocal_circuit(n_qubits, param_layers)
        else:
            assert n_ancilla_bits > 0, "Must specify ancilla bits for bayes"
            assert n_ancilla_reps > 0, "Must specify ancilla reps for bayes"
            self.ansatz = get_bqn_circuit(n_qubits, n_data_qubits, n_ancilla_bits,
                                         param_layers, n_ancilla_reps)

        # Store the original parameter objects in order before any transpilation
        # This acts as the source of truth for value mapping
        self.original_feature_params = list(self.feature_map.parameters)
        self.original_ansatz_params = list(self.ansatz.parameters)

        # Statevector simulator backend
        # Limit maximum parallel threads to one to avoid OpenMP deadlock
        self.backend = AerSimulator(method='statevector', max_parallel_threads=1)

        # Full circuit without measurements
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

        # Add statevector save instruction
        self.circuit.save_statevector()
        
        # Transpile the circuit for the simulator
        self.circuit = transpile(self.circuit, self.backend)

        self.num_qubits = n_qubits

        # Variational parameters
        n_weight_params = len(self.ansatz.parameters)
        self.weight_params = nn.Parameter(torch.randn(n_weight_params, dtype=torch.float32))

        # Output layer
        if qnn_output == 'trunc':
            self.output_layer = TruncateOutputLayer(n_actions)
        else:
            self.output_layer = nn.Linear(2 ** self.num_qubits, n_actions)

        # Parameter lists
        # Store names instead of objects since transpilation breaks object identity
        self.feature_param_names = [p.name for p in self.feature_map.parameters]
        self.ansatz_param_names = [p.name for p in self.ansatz.parameters]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode state, bind parameters, execute statevector simulation,
        extract probabilities, and pass through output layer.
        """
        # Encode the state
        x = self.encoder(x)
        x = x.float()

        input_np = x.detach().cpu().numpy()
        batch_size = input_np.shape[0]

        weight_np = self.weight_params.detach().cpu().numpy().flatten()
        
        # Get the exact parameters expected by the transpiled circuit
        circuit_parameters = list(self.circuit.parameters)

        binds = []
        for i in range(batch_size):
            param_dict = {}
            
            # Iterate through the parameters currently required by the circuit
            for param in circuit_parameters:
                p_name = param.name
                
                # Check if this parameter name exists in our original feature map
                if p_name in self.feature_param_names:
                    # Find its original index
                    original_idx = self.feature_param_names.index(p_name)
                    if original_idx < input_np.shape[1]:
                        param_dict[param] = float(input_np[i][original_idx])
                    else:
                        param_dict[param] = 0.0
                        
                # Check if this parameter name exists in our original ansatz
                elif p_name in self.ansatz_param_names:
                    # Find its original index
                    original_idx = self.ansatz_param_names.index(p_name)
                    if original_idx < len(weight_np):
                        param_dict[param] = float(weight_np[original_idx])
                    else:
                        param_dict[param] = 0.0
                else:
                    # If the transpiler introduced a completely new parameter default to zero
                    param_dict[param] = 0.0

            binds.append(param_dict)

        # Vectorized assignment via a list of dictionaries
        bound_circuits = self.circuit.assign_parameters(binds)

        # Execute all circuits of the batch in a single job 
        job = self.backend.run(bound_circuits)
        result = job.result()

        # Retrieve the probabilities for each simulated circuit
        probs_list = []
        for i in range(batch_size):
            # Get the statevector corresponding to the current index
            statevector = result.get_statevector(i)
            probs = np.abs(statevector.data) ** 2
            probs_list.append(probs)

        # Convert to torch tensor
        probs_np = np.array(probs_list)
        probs_tensor = torch.tensor(probs_np, dtype=torch.float32, device=x.device)

        # Apply output layer
        return self.output_layer(probs_tensor)
