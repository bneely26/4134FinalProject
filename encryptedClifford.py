from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile
from math import pi, sqrt
import random

def oracle_101(qc):
    """Oracle marking |101⟩ using multi-controlled Z gate."""
    qc.x(1)                 # Flip q1: |101⟩ → |111⟩
    qc.h(2)                 # Convert CCX to controlled-Z
    qc.ccx(0, 1, 2)         # Multi-controlled NOT
    qc.h(2)                 
    qc.x(1)                 # Restore q1

def diffuser(qc):
    qc.h(range(3))         
    qc.x(range(3))          
    qc.h(2)                 
    qc.ccx(0, 1, 2)        
    qc.h(2)
    qc.x(range(3))          
    qc.h(range(3))

def apply_qotp_encryption(qc, qubits, key_x, key_z):
    """
    Apply Quantum One-Time Pad encryption.
    
    Args:
        qc: Quantum circuit
        qubits: List of qubit indices to encrypt
        key_x: Bit-flip key (list of 0s and 1s)
        key_z: Phase-flip key (list of 0s and 1s)
    """
    for i, qubit in enumerate(qubits):
        if key_x[i] == 1:
            qc.x(qubit)
        if key_z[i] == 1:
            qc.z(qubit)

def decrypt_classical(measured_bits, key_x):
    """
    Decrypt the classical measurement result using the X key.
    
    Args:
        measured_bits: String of measured bits (e.g., '101')
        key_x: Bit-flip key used for encryption
    
    Returns:
        Decrypted bit string
    """
    decrypted = ""
    for i, bit in enumerate(measured_bits):
        decrypted += str(int(bit) ^ key_x[i])
    return decrypted

# Create Grover circuit for N=8, m=1
N = 8
n = 3  
optimal_iterations = int(pi/4 * sqrt(N))

qc = QuantumCircuit(3, 3)

# Initialize superposition
qc.h(range(3))

# Grover iteration (2 in our case)
for i in range(optimal_iterations):
    oracle_101(qc)
    diffuser(qc)

# ===== QUANTUM ONE-TIME PAD ENCRYPTION =====
# Generate random encryption keys
key_x = [random.randint(0, 1) for _ in range(n)]  # Bit-flip key
key_z = [random.randint(0, 1) for _ in range(n)]  # Phase-flip key

print("\n" + "="*50)
print("QUANTUM ONE-TIME PAD ENCRYPTION")
print("="*50)
print(f"X-key (bit-flip):  {key_x}")
print(f"Z-key (phase-flip): {key_z}")

# Apply QOTP encryption
qc.barrier()
apply_qotp_encryption(qc, range(3), key_x, key_z)
qc.barrier()

# Measurement
qc.measure(range(3), list(reversed(range(3))))

# Running the simulation
backend = Aer.get_backend("aer_simulator")
qc_t = transpile(qc, backend)
result = backend.run(qc_t, shots=1024).result()
counts = result.get_counts()

# Decrypt the results
decrypted_counts = {}
for measured, count in counts.items():
    decrypted = decrypt_classical(measured, key_x)
    decrypted_counts[decrypted] = decrypted_counts.get(decrypted, 0) + count

print(f"Decrypted counts: {decrypted_counts}")
print(f"Success probability for |101⟩: {decrypted_counts.get('101', 0)/1024:.1%}")

print(qc.draw("text"))