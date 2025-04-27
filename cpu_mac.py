# cpu_mac.py

class PipelineCPU:
    def __init__(self, program, memory_size=64):
        self.pc = 0
        self.reg = [0] * 32
        self.memory = [0] * memory_size
        self.program = program
        self.stages = [None] * 5  # [Fetch, Decode, Execute, Memory, Writeback]
        self.cycle = 0

    def get_register_value(self, reg_num, current_stage):
        # Check if the register is being written by a previous stage
        for stage in range(current_stage + 1, 5):
            if self.stages[stage] and self.stages[stage]['op'] in ['add', 'mul', 'lw']:
                if self.stages[stage]['rd'] == reg_num:
                    return self.stages[stage]['result']
        return self.reg[reg_num]

    def step(self):
        self.cycle += 1
        print(f"\nCycle {self.cycle}")
        print(f"PC: {self.pc}")
        print(f"Stages: {[str(stage) if stage else 'None' for stage in self.stages]}")
        print(f"Registers: {self.reg[1:12]}")  # Print relevant registers

        # Writeback Stage
        wb = self.stages[4]
        if wb:
            if wb['op'] in ['add', 'mul', 'lw']:
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

        # Fetch Stage
        if self.pc < len(self.program):
            self.stages[1] = self.program[self.pc]
            print(f"Fetch: {self.program[self.pc]['op']} instruction")
            self.pc += 1
        else:
            self.stages[1] = None

        # Clear Fetch stage for next cycle
        self.stages[0] = None

    def run(self):
        # Run until all instructions have completed the pipeline
        while any(stage is not None for stage in self.stages) or self.pc < len(self.program):
            self.step()