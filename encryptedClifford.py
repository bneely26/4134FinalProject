import random
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile
from math import pi, sqrt

# ----------------------------
# Random QOTP key generation
# ----------------------------
def random_qotp_keys(n):
    x = [random.randint(0,1) for _ in range(n)]
    z = [random.randint(0,1) for _ in range(n)]
    return x, z

def apply_qotp(qc, x, z):
    for i in range(len(x)):
        if x[i] == 1:
            qc.x(i)
        if z[i] == 1:
            qc.z(i)

# ----------------------------
# Your Grover oracle + diffuser
# ----------------------------
def oracle_101(qc):
    qc.x(1)
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(1)

def diffuser(qc):
    qc.h(range(3))
    qc.x(range(3))
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(range(3))
    qc.h(range(3))

# ----------------------------
# Build encrypted Grover circuit
# ----------------------------
N = 8
n = 3
iters = int(pi/4 * sqrt(N))

qc = QuantumCircuit(n, n)

# 1. Generate random QOTP keys
x_keys, z_keys = random_qotp_keys(n)

# 2. Encrypt initial state
apply_qotp(qc, x_keys, z_keys)

# 3. Grover iterations
qc.h(range(n))
for _ in range(iters):
    oracle_101(qc)
    diffuser(qc)

# 4. Decrypt before measurement: apply inverse pad
# inverse of X^x Z^z is Z^z X^x
for i in range(n):
    if z_keys[i] == 1:
        qc.z(i)
    if x_keys[i] == 1:
        qc.x(i)

# 5. Measure
qc.measure(range(n), list(reversed(range(n))))

# ----------------------------
# Run simulation
# ----------------------------
backend = Aer.get_backend("aer_simulator")
qc_t = transpile(qc, backend)
result = backend.run(qc_t, shots=1024).result()
counts = result.get_counts()

print("Encrypted Grover counts:", counts)
