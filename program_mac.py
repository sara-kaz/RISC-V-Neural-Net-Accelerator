# program_mac.py

from cpu_mac import PipelineCPU
import numpy as np

# Neural Network parameters
input_size = 3   # Number of input neurons
hidden_size = 2  # Number of hidden neurons
output_size = 1  # Number of output neurons

# Define a simple program to compute a 2-layer MLP with ReLU activation
program = [
    # Load inputs
    {'op': 'lw', 'rd': 1, 'addr': 0},    # input[0] into r1
    {'op': 'lw', 'rd': 2, 'addr': 1},    # input[1] into r2
    {'op': 'lw', 'rd': 3, 'addr': 2},    # input[2] into r3

    # Load weights for the hidden layer (input -> hidden)
    {'op': 'lw', 'rd': 4, 'addr': 10},   # weight[0,0] into r4
    {'op': 'lw', 'rd': 5, 'addr': 11},   # weight[0,1] into r5
    {'op': 'lw', 'rd': 6, 'addr': 12},   # weight[1,0] into r6
    {'op': 'lw', 'rd': 7, 'addr': 13},   # weight[1,1] into r7
    {'op': 'lw', 'rd': 8, 'addr': 14},   # bias[0] into r8
    {'op': 'lw', 'rd': 9, 'addr': 15},   # bias[1] into r9

    # Compute hidden layer: Z = W * X + b
    {'op': 'mul', 'rd': 10, 'rs1': 1, 'rs2': 4},  # r10 = input[0] × weight[0,0]
    {'op': 'mul', 'rd': 11, 'rs1': 2, 'rs2': 5},  # r11 = input[1] × weight[0,1]
    {'op': 'mul', 'rd': 12, 'rs1': 3, 'rs2': 6},  # r12 = input[2] × weight[1,0]
    {'op': 'mul', 'rd': 13, 'rs1': 2, 'rs2': 7},  # r13 = input[1] × weight[1,1]

    {'op': 'add', 'rd': 14, 'rs1': 10, 'rs2': 11}, # r14 = r10 + r11
    {'op': 'add', 'rd': 15, 'rs1': 12, 'rs2': 13}, # r15 = r12 + r13
    {'op': 'add', 'rd': 16, 'rs1': 14, 'rs2': 8},  # r16 = r14 + bias[0]
    {'op': 'add', 'rd': 17, 'rs1': 15, 'rs2': 9},  # r17 = r15 + bias[1]

    # Apply ReLU activation: ReLU(x) = max(0, x)
    {'op': 'max', 'rd': 18, 'rs1': 16, 'rs2': 0},  # ReLU for neuron 0 (hidden layer)
    {'op': 'max', 'rd': 19, 'rs1': 17, 'rs2': 0},  # ReLU for neuron 1 (hidden layer)

    # Load weights for the output layer (hidden -> output)
    {'op': 'lw', 'rd': 20, 'addr': 20},   # weight[0] into r20 (hidden->output)
    {'op': 'lw', 'rd': 21, 'addr': 21},   # weight[1] into r21 (hidden->output)
    {'op': 'lw', 'rd': 22, 'addr': 22},   # bias[2] into r22

    # Compute output layer: O = W * H + b
    {'op': 'mul', 'rd': 23, 'rs1': 18, 'rs2': 20},  # r23 = hidden[0] × weight[0]
    {'op': 'mul', 'rd': 24, 'rs1': 19, 'rs2': 21},  # r24 = hidden[1] × weight[1]

    {'op': 'add', 'rd': 25, 'rs1': 23, 'rs2': 24},  # r25 = r23 + r24
    {'op': 'add', 'rd': 26, 'rs1': 25, 'rs2': 22},  # r26 = r25 + bias[2]

    # Store result (output)
    {'op': 'sw', 'addr': 30, 'rs2': 26},
]

cpu = PipelineCPU(program)

# Initialize input values, weights, and biases
cpu.memory[0] = 1   # input[0]
cpu.memory[1] = 2   # input[1]
cpu.memory[2] = 3   # input[2]

cpu.memory[10] = 0.5 # weight[0,0]
cpu.memory[11] = 0.2 # weight[0,1]
cpu.memory[12] = 0.3 # weight[1,0]
cpu.memory[13] = 0.6 # weight[1,1]
cpu.memory[14] = 0.1 # bias[0]
cpu.memory[15] = 0.2 # bias[1]

cpu.memory[20] = 0.4 # weight[0] for output layer
cpu.memory[21] = 0.7 # weight[1] for output layer
cpu.memory[22] = 0.3 # bias[2] for output layer

cpu.run()

print(f"Neural Network Output (stored at memory[30]): {cpu.memory[30]}")
