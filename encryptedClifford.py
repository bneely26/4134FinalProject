from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import matplotlib.pyplot as plt
import random

def grover():
    qreg = QuantumRegister(3, "q")
    qc = QuantumCircuit(qreg)
    qc.h(qreg)
    # Oracle
    qc.x(qreg[1])
    qc.h(qreg[2])
    # CCX
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
    #Undo
    qc.h(qreg[2])
    qc.x(qreg[1])
    #Diffuser
    qc.h(qreg)
    qc.x(qreg)
    qc.h(qreg[2])
    # CCX
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
    #Undo
    qc.h(qreg[2])
    qc.x(qreg)
    qc.h(qreg)

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

def create_noise_model():
    noise_model = NoiseModel()
    
    p_gate1 = 0.001
    p_gate2 = 0.01 
    
    # Depolarizing errors for gates
    error_gate1 = depolarizing_error(p_gate1, 1)
    error_gate2 = depolarizing_error(p_gate2, 2)
    
    # Add errors to single-qubit gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ['h', 'x', 't', 'tdg'])
    
    # Add errors to two-qubit gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])
    
    # Readout error (measurement errors)
    # Probability of flipping 0->1 and 1->0
    p_meas = 0.02
    readout_error = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model

def main():
    backend = Aer.get_backend("aer_simulator")
    shots = 4096
    
    # Generate circuit and keys
    circuit = grover()
    a0 = [random.randint(0, 1) for _ in range(3)]
    b0 = [random.randint(0, 1) for _ in range(3)]
    
    print("SECTION 0: KEYS")
    print("\nInitial keys:")
    print("  a0:", a0)
    print("  b0:", b0)
    
    afinal, bfinal, _ = Compute_Keys(circuit, a0, b0)
    
    print("\nFinal keys:")
    print("  afinal:", afinal)
    print("  bfinal:", bfinal)
    
    # Create measurement circuit
    meas = QuantumCircuit(3, 3)
    meas.compose(circuit, inplace=True)
    meas.measure([0, 1, 2], [0, 1, 2])
    
    
    print("\nSECTION 1: IDEAL SIMULATION (NO NOISE)")    
    result_ideal = backend.run(meas, shots=shots).result()
    plain_counts_ideal = result_ideal.get_counts()
    encrypted_ideal = xor_results(plain_counts_ideal, afinal)
    decrypted_ideal = xor_results(encrypted_ideal, afinal)
    
    print_counts("Encrypted results (ideal):", encrypted_ideal)
    print_counts("Decrypted results (ideal):", decrypted_ideal)
    
    print("\nSECTION 2: NOISY SIMULATION")
    
    # Create noise model
    noise_model = create_noise_model()
    
    print("\nNoise model parameters:")
    print("  Single-qubit gate error: 0.1%")
    print("  Two-qubit gate error: 1.0%")
    print("  Readout error: 2.0%")
    
    # Run noisy simulation
    result_noisy = backend.run(meas, shots=shots, noise_model=noise_model).result()
    plain_counts_noisy = result_noisy.get_counts()
    encrypted_noisy = xor_results(plain_counts_noisy, afinal)
    decrypted_noisy = xor_results(encrypted_noisy, afinal)
    
    print_counts("Encrypted results (noisy):", encrypted_noisy)
    print_counts("Decrypted results (noisy):", decrypted_noisy)
    
    # Calculate fidelity metrics
    target_state = max(decrypted_ideal, key=decrypted_ideal.get)
    ideal_prob = decrypted_ideal[target_state] / shots
    noisy_prob = decrypted_noisy.get(target_state, 0) / shots
    
    print("\nNoise Analysis")
    print(f"Target state: {target_state}")
    print(f"Ideal probability: {ideal_prob} ({decrypted_ideal[target_state]}/{shots})")
    print(f"Noisy probability: {noisy_prob} ({decrypted_noisy.get(target_state, 0)}/{shots})")
    print(f"Fidelity loss: {(ideal_prob - noisy_prob)} ({((ideal_prob - noisy_prob)/ideal_prob * 100)}%)")

    # Plot simple bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ideal vs Noisy Results')
    
    ax1.bar(decrypted_ideal.keys(), decrypted_ideal.values())
    ax1.set_title('Ideal')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Count')
    
    ax2.bar(decrypted_noisy.keys(), decrypted_noisy.values())
    ax2.set_title('Noisy')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()