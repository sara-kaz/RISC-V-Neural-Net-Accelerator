# program_mac.py

from cpu_mac import PipelineCPU

# Define a simple program: compute a MAC (Multiply-Accumulate)
program = [
    # Load inputs
    {'op': 'lw', 'rd': 1, 'addr': 0},    # input[0] into r1
    {'op': 'lw', 'rd': 2, 'addr': 1},    # input[1] into r2
    {'op': 'lw', 'rd': 3, 'addr': 2},    # input[2] into r3

    # Load weights
    {'op': 'lw', 'rd': 4, 'addr': 10},   # weight[0] into r4
    {'op': 'lw', 'rd': 5, 'addr': 11},   # weight[1] into r5
    {'op': 'lw', 'rd': 6, 'addr': 12},   # weight[2] into r6

    # Multiply-accumulate
    {'op': 'mul', 'rd': 7, 'rs1': 1, 'rs2': 4},  # r7 = input[0] × weight[0]
    {'op': 'mul', 'rd': 8, 'rs1': 2, 'rs2': 5},  # r8 = input[1] × weight[1]
    {'op': 'mul', 'rd': 9, 'rs1': 3, 'rs2': 6},  # r9 = input[2] × weight[2]

    {'op': 'add', 'rd': 10, 'rs1': 7, 'rs2': 8}, # r10 = r7 + r8
    {'op': 'add', 'rd': 11, 'rs1': 10, 'rs2': 9},# r11 = r10 + r9

    # Store result
    {'op': 'sw', 'addr': 20, 'rs2': 11},
]

cpu = PipelineCPU(program)

# Initialize input values and weights
cpu.memory[0] = 2    # input 0
cpu.memory[1] = 3    # input 1
cpu.memory[2] = 1    # input 2

cpu.memory[10] = 5   # weight 0
cpu.memory[11] = 4   # weight 1
cpu.memory[12] = 6   # weight 2

cpu.run()

print(f"Output value (stored at memory[20]): {cpu.memory[20]}")