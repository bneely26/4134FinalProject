from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error
import matplotlib.pyplot as plt
import random
import math
import warnings
import pandas as pd
warnings.filterwarnings('ignore')



def grover_iteration(qc, qreg):
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

    # Undo oracle prep
    qc.h(qreg[2])
    qc.x(qreg[1])

    # Diffuser
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

    # Undo diffuser prep
    qc.h(qreg[2])
    qc.x(qreg)
    qc.h(qreg)

def grover():
    qreg = QuantumRegister(3, "q")
    qc = QuantumCircuit(qreg)
    
    # N and M are subject to changed based on the total number of items (N)
    # and how many items you are looking for (M)
    N = 8
    M = 1
    ideal_runs = math.floor((math.pi/4) * math.sqrt(N/M))

    # Initial superposition
    qc.h(qreg)

    for i in range(ideal_runs):
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

def create_noise_model():
    """Creates noise model with only depolarizing and readout errors"""
    noise_model = NoiseModel()

    # Error rates
    p_gate1 = 0.001  # .1% for single qubit gates
    p_gate2 = 0.01   # 1% for multi qubit gates
    p_meas = 0.02    # 2% error for readout

    # Create depolarizing errors
    error_gate1 = depolarizing_error(p_gate1, 1)
    error_gate2 = depolarizing_error(p_gate2, 2)

    # Add depolarizing errors to single qubit gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ['h', 'x', 't', 'tdg'])
    
    # Add depolarizing errors to two-qubit gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])
    
    # Adding readout error on measurement
    readout_error = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model
    
def create_plots(decrypted_ideal, decrypted_noisy):
    ## The following method was created with the help of AI ##
    ## All of the code has been reviewed for errors ##

    all_data = pd.read_csv("newData.csv")

    ideal = all_data.iloc[:, 0]
    noisy = all_data.iloc[:, 1]
    fidelity = all_data.iloc[:, 2]

    df = pd.DataFrame({
        "run": range(1, len(ideal) + 1),
        "Ideal": ideal,
        "Noisy": noisy,
        "Fidelity": fidelity
    })

    # Global research-style formatting
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "figure.titlesize": 20,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.5,
    })

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Ideal vs Noisy Results", fontsize=18, fontweight="normal")

    # ideal chart
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(list(decrypted_ideal.keys()), list(decrypted_ideal.values()), color='black')
    ax1.set_title("Ideal")
    ax1.set_xlabel("State")
    ax1.set_ylabel("Counts")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # noisy chart
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(list(decrypted_noisy.keys()), list(decrypted_noisy.values()), color='black')
    ax2.set_title("Noisy")
    ax2.set_xlabel("State")
    ax2.set_ylabel("Counts")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ideal, noisy, fidelity chart
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(df["run"], df["Ideal"], linestyle=':', label="Ideal", linewidth=2, color='black')
    ax3.plot(df["run"], df["Noisy"], linestyle='--', label="Noisy", linewidth=2, color='black')
    ax3.plot(df["run"], df["Fidelity"], label="Fidelity", linewidth=2, color='black')

    ax3.set_xlabel("Run Number")
    ax3.set_ylabel("Probability of Success")

    # legend
    ax3.legend(frameon=False, title=None)

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

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

    # create and show plots
    create_plots(decrypted_ideal, decrypted_noisy)

if __name__ == "__main__":
    main()