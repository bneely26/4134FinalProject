from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile
from math import pi, sqrt

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

# Create Grover circuit for N=8, m=1
N = 8
n = 3  
optimal_iterations = int(pi/4 * sqrt(N))

qc = QuantumCircuit(3, 3)

qc.h(range(3))

# Grover iteration (2 in our case)
for i in range(optimal_iterations):
    oracle_101(qc)
    diffuser(qc)

# Measurement
qc.measure(range(3), list(reversed(range(3))))

# Simulate
backend = Aer.get_backend("aer_simulator")
qc_t = transpile(qc, backend)
result = backend.run(qc_t, shots=1024).result()
counts = result.get_counts()

print(f"Grover search for N={N}, target=|101⟩:")
print(f"Success probability: {counts.get('101', 0)/1024:.1%}")
print(f"Counts: {counts}")
print(qc.draw("text"))

