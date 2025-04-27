# RISC-V Neural Net Accelerator

## Overview

This Python program simulates a basic 5-stage pipeline of a CPU, designed to accelerate neural network computations. The pipeline stages are as follows:

1. **Fetch**: The instruction at the current Program Counter (PC) is fetched and passed to the Decode stage.
2. **Decode**: The instruction is decoded, and the corresponding registers are prepared for execution.
3. **Execute**: The specified operation (addition, multiplication, max, etc.) is executed.
4. **Memory**: Memory operations (load or store) are performed in this stage.
5. **Writeback**: The result of the operation is written back to the appropriate register.

![Pipeline](https://www.researchgate.net/profile/Sajjad-Ahmed-23/publication/355051535/figure/fig3/AS:1076240400289792@1633607115859/The-RISC-V-ISA-compliant-RV32IM-5-Stage-fully-pipelined-datapath-designed-from-scratch.ppm)

The simulator tracks the program counter (PC), registers, memory, and the state of each pipeline stage. It also tracks the historical values of registers and provides a graphical plot of register activity over time.

## Features

- **Color-coded pipeline stages**: Each stage in the pipeline is color-coded for easy identification.
- **Register tracking**: Registers from `r1` to `r26` are tracked, including input layer, hidden layer weights, computations, and output layer weights and computations.
- **Cycle-by-cycle simulation**: The program runs through each cycle, printing the state of the pipeline and registers.
- **Graphical output**: After the simulation, a plot of register history is displayed using `matplotlib`.

## Neural Network Integration

The program is designed to simulate operations typically found in a neural network’s forward pass. It models the flow of computations and data processing through the layers of a neural network, using a simplified pipeline to represent each stage of computation.

### Neural Network Layer Structure
- **Input Layer Registers (r1 - r3)**: Represent the input features to the neural network.
- **Hidden Layer Weights (r4 - r9)**: Represent the weights of the neurons in the hidden layer.
- **Hidden Layer Computations (r10 - r17)**: Store the intermediate results of the neuron activations in the hidden layer.
- **ReLU Activations (r18 - r19)**: Store the outputs of the ReLU activation function applied to the hidden layer computations.
- **Output Layer Weights (r20 - r22)**: Represent the weights of the neurons in the output layer.
- **Output Computations (r23 - r26)**: Store the final computed outputs after applying the weights to the hidden layer results.
![Neural Network](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/3a/b8/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork.png)
### Connection to Neural Network

The simulation models the basic operations of a neural network, including matrix multiplication (for weighted sums), activation functions (e.g., ReLU), and the use of registers to hold intermediate computations. These computations are sequentially processed through the pipeline stages, providing a cycle-by-cycle visualization of how data flows through the network during the forward pass.

In essence, the CPU simulates how a neural network performs by computing activations layer by layer, storing the results in registers. The pipeline mimics the layers of a neural network, and the program provides a step-by-step visualization of this process, which is useful for debugging and understanding the computational flow in neural networks.

## Requirements

- Python 3.x
- `matplotlib`: For plotting register values over time.
- `termcolor`: For colored terminal output.

You can install the required dependencies using pip:

```bash
pip install matplotlib termcolor
```

## Usage

### CPU Initialization

The PipelineCPU class is initialized with:
- program: A list of instructions that the CPU will execute.
- memory_size: The size of memory, defaulting to 64.

The CPU begins execution with the program counter (PC) set to 0 and initializes the stages of the pipeline to None.
### Initialize CPU with the program
```bash
cpu = PipelineCPU(program)
```
### Run the CPU simulation
```bash
cpu.run()
```
### Methods
- get_register_value(reg_num, current_stage): Retrieves the value of a register, considering the current pipeline stage.
- color_stage(stage, name): Colors the pipeline stage for easy visualization in the terminal.
- step(): Simulates one cycle of the pipeline, progressing each stage and updating the register values.
- plot_registers(): Plots the history of register values over time using matplotlib.
- run(): Runs the full simulation, step by step, until all instructions have completed.

### Visualization

After running the program, the CPU visualizes the evolution of the registers. The following layers are visualized in a 3x2 grid of plots:
1. Input Layer Registers (r1 - r3)
2. Hidden Layer Weights (r4 - r9)
3. Hidden Layer Computations (r10 - r17)
4. ReLU Activations (r18 - r19)
5. Output Layer Weights (r20 - r22)
6. Output Computations (r23 - r26)

### Example program
```bash
program = [
    {'op': 'add', 'rs1': 1, 'rs2': 2, 'rd': 3},
    {'op': 'mul', 'rs1': 3, 'rs2': 4, 'rd': 5},
    {'op': 'max', 'rs1': 5, 'rs2': 6, 'rd': 7},
    {'op': 'lw', 'rs1': 7, 'rd': 8, 'addr': 0},
    {'op': 'sw', 'rs1': 8, 'rs2': 9, 'addr': 1}
]
```
### Example Output
```bash
============================================================
Cycle 1
PC: 0
------------------------------------------------------------
Stage      | Instruction                   
------------------------------------------------------------
Fetch      | add                         
Decode     | None                        
Execute    | None                        
Memory     | None                        
Writeback  | None                        
------------------------------------------------------------
Input Layer (r1-r3): [0, 0, 0]
Hidden Layer Weights (r4-r9): [0, 0, 0, 0, 0, 0]
Hidden Layer Computations (r10-r17): [0, 0, 0, 0, 0, 0, 0, 0]
ReLU Activations (r18-r19): [0, 0]
Output Layer Weights (r20-r22): [0, 0, 0]
Output Computations (r23-r26): [0, 0, 0, 0]
============================================================
Fetch: add instruction
```
### Instruction Format

Each instruction is represented as a dictionary containing:
- 'op': The operation (e.g., 'add', 'mul', 'max', 'lw', 'sw').
- 'rs1', 'rs2': Source registers.
- 'rd': Destination register.
- 'addr': Memory address for load/store operations (only applicable for lw and sw).

###  Register and Memory
- Registers are initialized to 0.
- Memory is also initialized to 0 with the size defined at initialization (default is 64).

###  Plotting

At the end of execution, a plot is generated that shows the evolution of the registers. The plot is divided into six subplots, each showing a different set of registers throughout the execution cycle.

