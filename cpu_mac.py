# cpu_mac.py

import time
import matplotlib.pyplot as plt
from termcolor import colored

class PipelineCPU:
    def __init__(self, program, memory_size=64):
        self.pc = 0
        self.reg = [0] * 32
        self.memory = [0] * memory_size
        self.program = program
        self.stages = [None] * 5  # [Fetch, Decode, Execute, Memory, Writeback]
        self.cycle = 0
        self.history = {i: [] for i in range(1, 27)}  # Track registers r1 to r26

    def get_register_value(self, reg_num, current_stage):
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
                print(f"Writeback: Writing {wb['result']} to r{wb['rd']}")
            self.stages[4] = None

        # Memory Stage
        mem = self.stages[3]
        if mem:
            if mem['op'] == 'lw':
                mem['result'] = self.memory[mem['addr']]
                print(f"Memory: Loading {mem['result']} from address {mem['addr']}")
            elif mem['op'] == 'sw':
                self.memory[mem['addr']] = self.reg[mem['rs2']]
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
                print(f"Execute: Adding r{exe['rs1']}({rs1_val}) + r{exe['rs2']}({rs2_val}) = {exe['result']}")
            elif exe['op'] == 'mul':
                rs1_val = self.get_register_value(exe['rs1'], 2)
                rs2_val = self.get_register_value(exe['rs2'], 2)
                exe['result'] = rs1_val * rs2_val
                print(f"Execute: Multiplying r{exe['rs1']}({rs1_val}) * r{exe['rs2']}({rs2_val}) = {exe['result']}")
            elif exe['op'] == 'max':
                rs1_val = self.get_register_value(exe['rs1'], 2)
                rs2_val = self.get_register_value(exe['rs2'], 2)
                exe['result'] = max(rs1_val, rs2_val)
                print(f"Execute: ReLU max(r{exe['rs1']}({rs1_val}), r{exe['rs2']}({rs2_val})) = {exe['result']}")
            self.stages[3] = exe
        else:
            self.stages[3] = None

        # Decode Stage
        dec = self.stages[1]
        if dec:
            print(f"Decode: {dec['op']} instruction")
            self.stages[2] = dec
        else:
            self.stages[2] = None

        # Move instruction from Fetch to Decode
        self.stages[1] = self.stages[0]

        # Fetch Stage
        if self.pc < len(self.program):
            self.stages[0] = self.program[self.pc]
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
        while any(stage is not None for stage in self.stages) or self.pc < len(self.program):
            self.step()
        
        self.plot_registers() 
        #time.sleep(0.2) # slow down between cycles like a real CPU simulator running
