from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import random

def grover_iteration(qc, qreg):
    # ----- Oracle -----
    qc.x(qreg[1])
    qc.h(qreg[2])

    # CCX (Toffoli decomposition)
    qc.h(qreg[2])
    qc.cx(qreg[1], qreg[2])
    qc.tdg(qreg[2])
    qc.cx(qreg[0], qreg[2])
    qc.t(qreg[2])
    qc.cx(qreg[1], qreg[2])
    qc.t(qreg[1])
    qc.tdg(qreg[2])
    qc.cx(qreg[0], qreg[2])
    qc.cx(qreg[0], qreg[1])
    qc.t(qreg[0])
    qc.tdg(qreg[1])
    qc.cx(qreg[0], qreg[1])
    qc.t(qreg[2])
    qc.h(qreg[2])

    # Undo oracle prep
    qc.h(qreg[2])
    qc.x(qreg[1])

    # ----- Diffuser -----
    qc.h(qreg)
    qc.x(qreg)
    qc.h(qreg[2])

    # CCX again
    qc.h(qreg[2])
    qc.cx(qreg[1], qreg[2])
    qc.tdg(qreg[2])
    qc.cx(qreg[0], qreg[2])
    qc.t(qreg[2])
    qc.cx(qreg[1], qreg[2])
    qc.t(qreg[1])
    qc.tdg(qreg[2])
    qc.cx(qreg[0], qreg[2])
    qc.cx(qreg[0], qreg[1])
    qc.t(qreg[0])
    qc.tdg(qreg[1])
    qc.cx(qreg[0], qreg[1])
    qc.t(qreg[2])
    qc.h(qreg[2])

    # Undo diffuser prep
    qc.h(qreg[2])
    qc.x(qreg)
    qc.h(qreg)

def grover():
    qreg = QuantumRegister(3, "q")
    qc = QuantumCircuit(qreg)

    # Initial superposition
    qc.h(qreg)

    # ---- TWO Grover iterations ----
    grover_iteration(qc, qreg)
    grover_iteration(qc, qreg)

    return qc

def KeyH(a, b, idx):
    a[idx], b[idx] = b[idx], a[idx]

def KeyCX(a, b, control, target):
    a[target] ^= a[control]
    b[control] ^= b[target]

def KeyT(a, b, idx, ra, rb):
    prev_a = a[idx]
    a[idx] ^= ra
    b[idx] ^= (prev_a ^ rb)

def KeyTdg(a, b, idx, ra, rb):
    a[idx] ^= ra
    b[idx] ^= rb

def key_helper(inst, qmap, a, b, tvals):
    opname = inst.operation.name.lower()
    qubits = inst.qubits
    if opname == "h":
        KeyH(a, b, qmap[qubits[0]])
    elif opname in ("cx", "cnot"):
        KeyCX(a, b, qmap[qubits[0]], qmap[qubits[1]])
    elif opname in ("t", "tdg"):
        ra, rb = next(tvals)
        idx = qmap[qubits[0]]
        if opname == "t":
            KeyT(a, b, idx, ra, rb)
        else:
            KeyTdg(a, b, idx, ra, rb)

def Compute_Keys(qc, a, b):
    qmap = {q: i for i, q in enumerate(qc.qubits)}
    t= sum(
        inst.operation.name.lower() in ("t", "tdg")
        for inst in qc.data
    )
    t_measurements = [(random.randint(0, 1), random.randint(0, 1)) for i in range(t)]
    t_measures = iter(t_measurements)
    for inst in qc.data:
        key_helper(inst, qmap, a, b, t_measures)
    return a, b, t_measurements

def xor(bitstring, a):
    reverse = bitstring[::-1]
    out = ''.join(str(int(b) ^ k) for b, k in zip(reverse, a))
    return out[::-1]

def xor_results(counts, a):
    out = {}
    for bits, num in counts.items():
        mapped = out.get(bits_x := xor(bits, a), 0)
        out[bits_x] = mapped + num
    return out

def print_counts(title, counts):
    print("\n" + title) 
    for s in sorted(counts.keys()):
        print(f"  {s}: {counts[s]}")

def main():
    backend = Aer.get_backend("aer_simulator")
    shots = 4096
    circuit = grover()
    a0 = [random.randint(0, 1) for _ in range(3)]
    b0 = [random.randint(0, 1) for _ in range(3)]
    print("\nInitial keys:")
    print("  a0:", a0)
    print("  b0:", b0)
    afinal, bfinal, _ = Compute_Keys(circuit, a0, b0)
    print("\nFinal keys:")
    print("  afinal:", afinal)
    print("  bfinal:", bfinal)
    meas = QuantumCircuit(3, 3)
    meas.compose(circuit, inplace=True)
    meas.measure([0, 1, 2], [0, 1, 2])
    result = backend.run(meas, shots=shots).result()
    plain_counts = result.get_counts()
    encrypted = xor_results(plain_counts, afinal)
    decrypted = xor_results(encrypted, afinal)
    print_counts("Encrypted results:", encrypted)
    print_counts("Decrypted results:", decrypted)


if __name__ == "__main__":
    main()
