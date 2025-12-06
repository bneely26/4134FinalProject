from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


# Grovers with decomposed CCX
qc = QuantumCircuit(3)

# Initial superposition
qc.h([0, 1, 2])
for i in range(2): 
#Phase Flip |101>
    qc.x(1)            
    qc.ccz(0, 1, 2)    
    qc.x(1)            

    # Diffuser
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    #Decomposed CCX
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.t(1)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.t(0)
    qc.tdg(1)
    qc.cx(0, 1)
    qc.t(2)
    #Diffuser
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])



# Measurement and Results
qc.measure_all()
backend = Aer.get_backend("aer_simulator")
qc_t = transpile(qc, backend)
result = backend.run(qc_t, shots=1024).result()
counts = result.get_counts()

print(counts)
print(f"Accuracy = {counts.get('101',0)/1024:.1%}")


print("\nCircuit:")
print(qc.draw("text"))
