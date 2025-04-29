# cpu_mac.py

import time
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PipelineCPU:
    def __init__(self, program=None, memory_size=1024, verbose=False):
        self.pc = 0
        self.reg = [0] * 32
        self.memory = [0] * memory_size
        self.program = program or []
        self.stages = [None] * 5  # [Fetch, Decode, Execute, Memory, Writeback]
        self.cycle = 0
        self.history = {i: [] for i in range(1, 27)}  # Track registers r1 to r26
        self.verbose = verbose

        # MNIST specific attributes
        self.input_size = 28 * 28  # MNIST image size
        self.hidden_size = 128
        self.output_size = 10  # 10 digits (0-9)
        self.weights_hidden = None
        self.weights_output = None
        self.bias_hidden = None
        self.bias_output = None

    def load_mnist_data(self):
        """Load and preprocess MNIST data"""
        print("Loading MNIST data...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
        X = X / 255.0  # Normalize to [0,1]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def initialize_weights(self):
        """Initialize weights for the neural network"""
        print("Initializing weights...")
        # Initialize weights with Xavier/Glorot initialization
        self.weights_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.weights_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / (self.hidden_size + self.output_size))

        # Initialize biases
        self.bias_hidden = np.zeros(self.hidden_size)
        self.bias_output = np.zeros(self.output_size)

        # Store weights in memory
        self.memory[0:self.input_size*self.hidden_size] = self.weights_hidden.flatten()
        self.memory[self.input_size*self.hidden_size:self.input_size*self.hidden_size + self.hidden_size*self.output_size] = self.weights_output.flatten()
        self.memory[self.input_size*self.hidden_size + self.hidden_size*self.output_size:self.input_size*self.hidden_size + self.hidden_size*self.output_size + self.hidden_size] = self.bias_hidden
        self.memory[self.input_size*self.hidden_size + self.hidden_size*self.output_size + self.hidden_size:self.input_size*self.hidden_size + self.hidden_size*self.output_size + self.hidden_size + self.output_size] = self.bias_output

    def predict(self, image):
        """Make a prediction for a single MNIST image"""
        # We'll use registers with limited range:
        # r1-r5: For temporary values and computation
        # r6-r9: For accumulating results

        # Load image into memory (not directly into registers)
        for i in range(len(image)):
            self.memory[800 + i] = image[i]

        program = []

        # Hidden layer computation
        for i in range(self.hidden_size):
            # Reset accumulator
            program.append({'op': 'add', 'rd': 6, 'rs1': 0, 'rs2': 0})  # r6 = 0

            # Compute dot product in chunks
            for j in range(self.input_size):
                # Load input pixel
                program.append({'op': 'lw', 'rd': 1, 'addr': 800 + j})
                # Load weight
                program.append({'op': 'lw', 'rd': 2, 'addr': i*self.input_size + j})
                # Multiply
                program.append({'op': 'mul', 'rd': 3, 'rs1': 1, 'rs2': 2})
                # Accumulate
                program.append({'op': 'add', 'rd': 6, 'rs1': 6, 'rs2': 3})

            # Add bias
            program.append({'op': 'lw', 'rd': 4, 'addr': self.input_size*self.hidden_size + self.hidden_size*self.output_size + i})
            program.append({'op': 'add', 'rd': 6, 'rs1': 6, 'rs2': 4})

            # ReLU activation
            program.append({'op': 'max', 'rd': 6, 'rs1': 6, 'rs2': 0})

            # Store result to memory
            program.append({'op': 'sw', 'addr': 900 + i, 'rs2': 6})

        # Output layer computation
        for i in range(self.output_size):
            # Reset accumulator
            program.append({'op': 'add', 'rd': 7, 'rs1': 0, 'rs2': 0})  # r7 = 0

            for j in range(self.hidden_size):
                # Load hidden value
                program.append({'op': 'lw', 'rd': 1, 'addr': 900 + j})
                # Load weight
                program.append({'op': 'lw', 'rd': 2, 'addr': self.input_size*self.hidden_size + j*self.output_size + i})
                # Multiply
                program.append({'op': 'mul', 'rd': 3, 'rs1': 1, 'rs2': 2})
                # Accumulate
                program.append({'op': 'add', 'rd': 7, 'rs1': 7, 'rs2': 3})

            # Add bias
            program.append({'op': 'lw', 'rd': 4, 'addr': self.input_size*self.hidden_size + self.hidden_size*self.output_size + self.hidden_size + i})
            program.append({'op': 'add', 'rd': 7, 'rs1': 7, 'rs2': 4})

            # Store result
            program.append({'op': 'sw', 'addr': 1000 + i, 'rs2': 7})

        self.program = program
        self.run()

        # Get prediction
        output = [self.memory[1000 + i] for i in range(self.output_size)]
        return np.argmax(output)

    def get_register_value(self, reg_num, current_stage):
       # Add bounds checking
       if reg_num >= len(self.reg):
           print(f"Error: Trying to access register r{reg_num} which doesn't exist")
           return 0  # Return a default value

       # Check if the register is being written by a previous stage
       for stage in range(current_stage + 1, 5):
           if self.stages[stage] and self.stages[stage]['op'] in ['add', 'mul', 'lw', 'max']:
               if self.stages[stage]['rd'] == reg_num:
                   return self.stages[stage]['result']
       return self.reg[reg_num]

    def color_stage(self, stage, name):
        if not stage:
            return colored('None', 'white')

        color_map = {
            'fetch': 'blue',
            'decode': 'cyan',
            'execute': 'green',
            'memory': 'yellow',
            'writeback': 'magenta'
        }

        return colored(f"{stage['op']}", color_map[name.lower()])

    def step(self):
        self.cycle += 1
        if self.verbose:
            print("\n" + "="*60)
            print(f"Cycle {self.cycle}")
            print(f"PC: {self.pc}")
            print("-"*60)

            # Color stages and print them
            stage_names = ['Fetch', 'Decode', 'Execute', 'Memory', 'Writeback']
            stages_colored = [self.color_stage(stage, name) if stage else 'None' for stage, name in zip(self.stages, stage_names)]
            print(f"{'Stage':<10} | {'Instruction':<30}")
            print("-"*60)
            for i, stage in enumerate(stages_colored):
                print(f"{stage_names[i]:<10} | {stage:<30}")

            print("-"*60)
            # Print relevant registers for neural network
            print("Input Layer (r1-r3):", self.reg[1:4])
            print("Hidden Layer Weights (r4-r9):", self.reg[4:10])
            print("Hidden Layer Computations (r10-r17):", self.reg[10:18])
            print("ReLU Activations (r18-r19):", self.reg[18:20])
            print("Output Layer Weights (r20-r22):", self.reg[20:23])
            print("Output Computations (r23-r26):", self.reg[23:27])
            print("="*60)

        # Writeback Stage
        wb = self.stages[4]
        if wb:
            if wb['op'] in ['add', 'mul', 'lw', 'max']:
                self.reg[wb['rd']] = wb['result']
                if self.verbose:
                    print(f"Writeback: Writing {wb['result']} to r{wb['rd']}")
            self.stages[4] = None

        # Memory Stage
        mem = self.stages[3]
        if mem:
            if mem['op'] == 'lw':
                mem['result'] = self.memory[mem['addr']]
                if self.verbose:
                    print(f"Memory: Loading {mem['result']} from address {mem['addr']}")
            elif mem['op'] == 'sw':
                self.memory[mem['addr']] = self.reg[mem['rs2']]
                if self.verbose:
                    print(f"Memory: Storing {self.reg[mem['rs2']]} to address {mem['addr']}")
            self.stages[4] = mem
        else:
            self.stages[4] = None

        # Execute Stage
        exe = self.stages[2]
        if exe:
            if exe['op'] == 'add':
                rs1_val = self.get_register_value(exe['rs1'], 2)
                rs2_val = self.get_register_value(exe['rs2'], 2)
                exe['result'] = rs1_val + rs2_val
                if self.verbose:
                    print(f"Execute: Adding r{exe['rs1']}({rs1_val}) + r{exe['rs2']}({rs2_val}) = {exe['result']}")
            elif exe['op'] == 'mul':
                rs1_val = self.get_register_value(exe['rs1'], 2)
                rs2_val = self.get_register_value(exe['rs2'], 2)
                exe['result'] = rs1_val * rs2_val
                if self.verbose:
                    print(f"Execute: Multiplying r{exe['rs1']}({rs1_val}) * r{exe['rs2']}({rs2_val}) = {exe['result']}")
            elif exe['op'] == 'max':
                rs1_val = self.get_register_value(exe['rs1'], 2)
                rs2_val = self.get_register_value(exe['rs2'], 2)
                exe['result'] = max(rs1_val, rs2_val)
                if self.verbose:
                    print(f"Execute: ReLU max(r{exe['rs1']}({rs1_val}), r{exe['rs2']}({rs2_val})) = {exe['result']}")
            self.stages[3] = exe
        else:
            self.stages[3] = None

        # Decode Stage
        dec = self.stages[1]
        if dec:
            if self.verbose:
                print(f"Decode: {dec['op']} instruction")
            self.stages[2] = dec
        else:
            self.stages[2] = None

        # Move instruction from Fetch to Decode
        self.stages[1] = self.stages[0]

        # Fetch Stage
        if self.pc < len(self.program):
            self.stages[0] = self.program[self.pc]
            if self.verbose:
                print(f"Fetch: {self.program[self.pc]['op']} instruction")
            self.pc += 1
        else:
            self.stages[0] = None

        for reg_num in self.history.keys():
            self.history[reg_num].append(self.reg[reg_num])

    def plot_registers(self):
        plt.figure(figsize=(15,10))

        # Plot input layer
        plt.subplot(3, 2, 1)
        for reg_num in range(1, 4):
            plt.plot(self.history[reg_num], label=f"r{reg_num}")
        plt.title("Input Layer Registers")
        plt.legend()

        # Plot hidden layer weights
        plt.subplot(3, 2, 2)
        for reg_num in range(4, 10):
            plt.plot(self.history[reg_num], label=f"r{reg_num}")
        plt.title("Hidden Layer Weights")
        plt.legend()

        # Plot hidden layer computations
        plt.subplot(3, 2, 3)
        for reg_num in range(10, 18):
            plt.plot(self.history[reg_num], label=f"r{reg_num}")
        plt.title("Hidden Layer Computations")
        plt.legend()

        # Plot ReLU activations
        plt.subplot(3, 2, 4)
        for reg_num in range(18, 20):
            plt.plot(self.history[reg_num], label=f"r{reg_num}")
        plt.title("ReLU Activations")
        plt.legend()

        # Plot output layer weights
        plt.subplot(3, 2, 5)
        for reg_num in range(20, 23):
            plt.plot(self.history[reg_num], label=f"r{reg_num}")
        plt.title("Output Layer Weights")
        plt.legend()

        # Plot output computations
        plt.subplot(3, 2, 6)
        for reg_num in range(23, 27):
            plt.plot(self.history[reg_num], label=f"r{reg_num}")
        plt.title("Output Computations")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run(self):
        # Run until all instructions have completed the pipeline
        total_instructions = len(self.program)
        while any(stage is not None for stage in self.stages) or self.pc < len(self.program):
            if not self.verbose and self.cycle % 1000 == 0:
                progress = min(100, (self.pc / total_instructions) * 100)
                print(f"\rProcessing: {progress:.1f}% complete", end="")
            self.step()

        if not self.verbose:
            print("\rProcessing: 100% complete")

        if self.verbose:
            self.plot_registers()

# Example usage
if __name__ == "__main__":
    # Create CPU instance
    cpu = PipelineCPU()

    # Load MNIST data
    X_train, X_test, y_train, y_test = cpu.load_mnist_data()

    # Initialize weights
    cpu.initialize_weights()

    # Make prediction on first test image
    test_image = X_test[0]
    prediction = cpu.predict(test_image)
    print(f"\nPredicted digit: {prediction}")
    print(f"Actual digit: {y_test[0]}")
