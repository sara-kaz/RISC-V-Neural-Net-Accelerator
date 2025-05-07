# Simulated CPU Pipeline for MNIST Digit Prediction

## Overview

This Python script implements a custom CPU pipeline simulator that mimics instruction execution across 5 stages (Fetch, Decode, Execute, Memory, Writeback) to perform predictions on the MNIST handwritten digit dataset using a simple neural network (1 hidden layer). The simulation models how a CPU would process instructions to carry out neural network inference, providing insight into low-level data flow and register usage.

<div align="center">
  
  ![MINST](https://upload.wikimedia.org/wikipedia/commons/b/b1/MNIST_dataset_example.png)
   
   MNIST Dataset View
</div>

## Features
- Fetches and preprocesses the MNIST dataset using scikit-learn.
- Initializes weights and biases using Xavier/Glorot initialization.
- Simulates a 5-stage pipeline CPU with a simple instruction set: add, mul, lw, sw, max.
- Tracks register history for visualization and debugging.
- Predicts digits using a manually assembled “program” of instructions for forward propagation.
- Includes ReLU activation and softmax-like decision via argmax.

## Requirements
- Python 3.7+
- numpy
- matplotlib
- termcolor
- scikit-learn

Install dependencies using:

```
pip install numpy matplotlib termcolor scikit-learn
```


## Files
- train.py: Main simulation script defining the PipelineCPU class and its methods.
- test.py: Testing a set number of MNIST images 
- (Optional) If you’re planning to visualize the register values or pipeline stages, you can extend the script with plotting features using matplotlib.

## Usage
1.	Import and instantiate the pipeline CPU:
   
```
from train import PipelineCPU
cpu = PipelineCPU(verbose=True)
```

2.	Load and prepare data:

```
X_train, X_test, y_train, y_test = cpu.load_mnist_data()
```
3.	Initialize neural network weights:
```
cpu.initialize_weights()
```
4.	Predict a digit:

```
sample = X_test[0]
prediction = cpu.predict(sample)
print("Predicted:", prediction)
```

## Pipeline Simulation Details

The CPU pipeline simulates the following stages:
1. Fetch: Retrieves instruction from program using the program counter (pc).
2. Decode: Parses operands and prepares for execution.
3. Execute: Performs arithmetic/logical operations.
4. Memory: Loads/stores data from/to simulated memory.
5. Writeback: Writes results back to registers.

Each register (r1 to r26) is tracked per cycle, allowing for detailed analysis.

## Instruction Set
- add: Add values from two registers.
- mul: Multiply values from two registers.
- lw: Load word from memory to register.
- sw: Store word from register to memory.
- max: ReLU activation simulated as max(0, x).
