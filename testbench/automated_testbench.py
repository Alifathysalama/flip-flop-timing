"""
AUTOMATED SELF-CHECKING FLIP-FLOP TIMING TESTBENCH
===================================================
Based on daniestevez/flip-flop-timing but with full automation and self-checking

Key Differences from Author's Code:
1. AUTOMATED: No manual plot inspection - automatic pass/fail decisions
2. SELF-CHECKING: Compares measurements against expected specifications
3. COMPREHENSIVE: Tests all timing parameters with one command
4. REGRESSION FRIENDLY: Returns exit codes for CI/CD integration
5. DETAILED REPORTING: Shows what passed/failed and why
6. REUSABLE: Class-based design for easy modification

Expected Results:
- Output Delay Fast: ~101ps (spec: 101ps) ✓
- Output Delay Slow: ~229ps or 325ps (spec: 325ps) ✓
- Setup Time: ~-70ps (spec: -70ps) ✓
- Hold Time: ~9ps (spec: 9ps) ✓
"""

import pickle
import subprocess
import tempfile
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

plt.rcParams['figure.figsize'] = (12, 8)


class TestStatus(Enum):
    """Test result status"""
    PASS = "✓ PASS"
    FAIL = "✗ FAIL"
    WARN = "⚠ WARN"
    INFO = "ℹ INFO"


@dataclass
class TimingSpecs:
    """Expected timing specifications"""
    output_delay_fast_min: float = 95e-12   # 95ps minimum
    output_delay_fast_max: float = 110e-12  # 110ps maximum
    output_delay_slow_min: float = 220e-12  # 220ps minimum  
    output_delay_slow_max: float = 350e-12  # 350ps maximum
    setup_time_target: float = -70e-12      # -70ps
    setup_time_tolerance: float = 10e-12    # ±10ps
    hold_time_target: float = 9e-12         # 9ps
    hold_time_tolerance: float = 3e-12      # ±3ps
    vdd_nominal: float = 1.2
    vdd_threshold_high: float = 0.7
    vdd_threshold_low: float = 0.3


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: TestStatus
    measured: float
    expected: float
    unit: str = "ps"
    message: str = ""
    
    def __str__(self):
        meas_str = f"{self.measured * 1e12:.1f}" if self.unit == "ps" else f"{self.measured:.3f}"
        exp_str = f"{self.expected * 1e12:.1f}" if self.unit == "ps" else f"{self.expected:.3f}"
        return f"{self.status.value}: {self.name} = {meas_str}{self.unit} (expected: {exp_str}{self.unit}) {self.message}"


class FlipFlopAutomatedTestbench:
    """Fully automated self-checking testbench"""
    
    def __init__(self, specs: TimingSpecs, save_plots: bool = True, verbose: bool = True):
        self.specs = specs
        self.save_plots = save_plots
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.vdd = specs.vdd_nominal
        self.clock_period = 1e-9
        self.temperature = 27
        
        # Data storage
        self.data_cache = {}
        
    # ==================== CORE SIMULATION FUNCTIONS ====================
    
    def run_ngspice(self, script: str) -> None:
        """Execute ngspice simulation"""
        with tempfile.NamedTemporaryFile('w', delete_on_close=False, suffix='.cir') as f:
            f.write(script)
            f.close()
            result = subprocess.run([
                '/bin/bash', '-c',
                f'cd ../; source setup-pdk; cd - > /dev/null; ngspice {f.name} > /dev/null 2>&1'
            ], capture_output=True)
            if result.returncode != 0:
                print(f"ERROR: ngspice failed: {result.stderr.decode()}")
                raise RuntimeError("ngspice simulation failed")
    
    def read_spice_output(self, filepath: str) -> Dict:
        """Parse ngspice output"""
        signals = ['C', 'R', 'D', 'nand0', 'nand1', 'nand2', 'nand3', 'Q', 'nand5']
        data = np.fromfile(filepath, sep=' ').reshape(-1, 2 * len(signals))
        t = {s: data[:, 2*j] for j, s in enumerate(signals)}
        x = {s: data[:, 2*j+1] for j, s in enumerate(signals)}
        return {'t': t, 'x': x}
    
    def run_sim(self, template: str, **kwargs) -> Dict:
        """Run single simulation"""
        with tempfile.NamedTemporaryFile('w', delete_on_close=False, suffix='.txt') as f_out:
            f_out.close()
            self.run_ngspice(template.format(output_file=f_out.name, **kwargs))
            return self.read_spice_output(f_out.name)
    
    def vdd_model(self) -> float:
        """Random VDD variation"""
        return np.random.uniform(self.vdd - 0.05, self.vdd + 0.05)
    
    def temperature_model(self) -> float:
        """Random temperature variation"""
        return np.random.uniform(-10, 100)
    
    # ==================== MEASUREMENT FUNCTIONS (from author) ====================
    
    def output_delay(self, traces: Dict) -> Dict[str, float]:
        """
        Measure output delay (author's exact algorithm)
        Returns fast and slow delays for both 0->1 and 1->0 transitions
        """
        vdd = traces.get('vdd', self.vdd)
        
        # 0->1 transition (slow = crossing 70% threshold)
        threshold_crosses = traces['t']['Q'][np.where(np.diff(traces['x']['Q'] > 0.7 * vdd))[0]]
        delay_01_slow = (
            np.max(threshold_crosses[(self.clock_period < threshold_crosses)
                & (threshold_crosses < 2 * self.clock_period)])
            - self.clock_period)
        
        # 0->1 transition (fast = crossing 30% threshold)
        threshold_crosses = traces['t']['Q'][np.where(np.diff(traces['x']['Q'] >= 0.3 * vdd))[0]]
        delay_01_fast = (
            np.min(threshold_crosses[(self.clock_period < threshold_crosses)
                & (threshold_crosses < 2 * self.clock_period)])
            - self.clock_period)
        
        # 1->0 transition (slow = crossing 30% threshold)
        threshold_crosses = traces['t']['Q'][np.where(np.diff(traces['x']['Q'] < 0.3 * vdd))[0]]
        delay_10_slow = (
            np.max(threshold_crosses[(2 * self.clock_period < threshold_crosses)
                & (threshold_crosses < 3 * self.clock_period)])
            - 2 * self.clock_period)
        
        # 1->0 transition (fast = crossing 70% threshold)
        threshold_crosses = traces['t']['Q'][np.where(np.diff(traces['x']['Q'] <= 0.7 * vdd))[0]]
        delay_10_fast = (
            np.min(threshold_crosses[(2 * self.clock_period < threshold_crosses)
                & (threshold_crosses < 3 * self.clock_period)])
            - 2 * self.clock_period)
        
        return {
            '01_fast': delay_01_fast,
            '01_slow': delay_01_slow,
            '10_fast': delay_10_fast,
            '10_slow': delay_10_slow,
            'fast': min(delay_01_fast, delay_10_fast),
            'slow': max(delay_01_slow, delay_10_slow),
        }
    
    def setup_cost(self, traces: Dict, setup: float) -> float:
        """Calculate setup cost metric (author's algorithm)"""
        vdd = traces.get('vdd', self.vdd)
        if traces.get('rising_edge', True):
            threshold_crosses = traces['t']['Q'][np.where(np.diff(traces['x']['Q'] > 0.7 * vdd))[0]]
        else:
            threshold_crosses = traces['t']['Q'][np.where(np.diff(traces['x']['Q'] < 0.3 * vdd))[0]]
        output_delay = np.max(threshold_crosses[(2 * self.clock_period < threshold_crosses)]) - 2 * self.clock_period
        return output_delay - setup
    
    def meets_hold(self, trace: Dict) -> bool:
        """Check if hold time is met (author's algorithm)"""
        sel = trace['t']['Q'] >= 2e-9 + 325e-12
        if trace.get('rising_edge', True):
            return np.all(trace['x']['Q'][sel] <= 0.3 * trace['vdd'])
        return np.all(trace['x']['Q'][sel] >= 0.7 * trace['vdd'])
    
    # ==================== TEST 1: OUTPUT DELAY ====================
    
    def test_output_delay(self) -> bool:
        """Test output delay across all corners with self-checking"""
        print("\n" + "="*70)
        print("TEST 1: OUTPUT DELAY CHARACTERIZATION")
        print("="*70)
        
        template = """
.temp {temperature}
.lib cornerMOSlv.lib {mos_corner}
.include ../magic/fdc_dense.spice

Vdd VDD GND {vdd}
Vss VSS GND 0
.param rise_time=0p
.param clock_period=1n
Vresetn R GND PULSE({vdd} 0 0p 0 rise_time {{0.5 * clock_period}} 0)
Vdata D GND PULSE(0 {vdd} {{0.5 * clock_period}} rise_time rise_time clock_period 0)
Vclk C GND PULSE(0 {vdd} clock_period rise_time rise_time {{0.5 * clock_period}} clock_period)

.tran 1p {{3 * clock_period}}

.control
run
wrdata {output_file} C R D nand0 nand1 nand2 nand3 Q nand5
quit
.endc
.end
"""
        
        corners = ['mos_tt_stat', 'mos_ff', 'mos_ss', 'mos_sf', 'mos_fs', 'mos_tt']
        runs_per_corner = {'mos_tt_stat': 100}
        
        # Try to load cached data
        try:
            data_output_delay = self.load_data('output_delay.pickle')
            print("  ℹ Loaded cached simulation data")
        except FileNotFoundError:
            print("  ℹ Running simulations (this may take a while)...")
            data_output_delay = {}
            for corner in corners:
                def sim(vdd=self.vdd, temperature=self.temperature):
                    return self.run_sim(template, mos_corner=corner, vdd=vdd, temperature=temperature) | {'vdd': vdd}
                
                if corner in runs_per_corner:
                    for run in range(runs_per_corner[corner]):
                        if self.verbose and run % 20 == 0:
                            print(f"    Running {corner} simulation {run}/{runs_per_corner[corner]}...")
                        data_output_delay[f'{corner}_{run:04}'] = sim(vdd=self.vdd_model(), temperature=self.temperature_model())
                else:
                    print(f"    Running {corner} simulation...")
                    data_output_delay[corner] = sim()
            self.save_data('output_delay.pickle', data_output_delay)
        
        self.data_cache['output_delay'] = data_output_delay
        
        # Calculate delays for all runs
        print("\n  Measuring output delays...")
        delays = {k: self.output_delay(v) for k, v in data_output_delay.items()}
        
        # Extract overall fast and slow
        output_delay_fast = min(d['fast'] for d in delays.values())
        output_delay_slow = max(d['slow'] for d in delays.values())
        
        print(f"\n  Measured across all corners and runs:")
        print(f"    Fast corner: {output_delay_fast*1e12:.1f} ps")
        print(f"    Slow corner: {output_delay_slow*1e12:.1f} ps")
        
        # Self-checking
        fast_pass = self.specs.output_delay_fast_min <= output_delay_fast <= self.specs.output_delay_fast_max
        slow_pass = self.specs.output_delay_slow_min <= output_delay_slow <= self.specs.output_delay_slow_max
        
        self.results.append(TestResult(
            "Output Delay (Fast Corner)",
            TestStatus.PASS if fast_pass else TestStatus.FAIL,
            output_delay_fast,
            (self.specs.output_delay_fast_min + self.specs.output_delay_fast_max) / 2,
            message=f"Range: [{self.specs.output_delay_fast_min*1e12:.1f}, {self.specs.output_delay_fast_max*1e12:.1f}]ps"
        ))
        
        self.results.append(TestResult(
            "Output Delay (Slow Corner)",
            TestStatus.PASS if slow_pass else TestStatus.FAIL,
            output_delay_slow,
            (self.specs.output_delay_slow_min + self.specs.output_delay_slow_max) / 2,
            message=f"Range: [{self.specs.output_delay_slow_min*1e12:.1f}, {self.specs.output_delay_slow_max*1e12:.1f}]ps"
        ))
        
        # Generate plot
        if self.save_plots:
            self.plot_output_delay(data_output_delay)
        
        return fast_pass and slow_pass
    
    # ==================== TEST 2: SETUP TIME ====================
    
    def test_setup_time(self) -> bool:
        """Test setup time with self-checking"""
        print("\n" + "="*70)
        print("TEST 2: SETUP TIME CHARACTERIZATION")
        print("="*70)
        
        template = """
.temp {temperature}
.lib cornerMOSlv.lib {mos_corner}
.include ../magic/fdc_dense.spice

Vdd VDD GND {vdd}
Vss VSS GND 0
.param rise_time=0p
.param clock_period=1n
Vresetn R GND {vdd}
Vdata D GND PULSE({data0} {data1} {{2 * clock_period + {data_change}}} 0 0 0)
Vclk C GND PULSE(0 {vdd} clock_period rise_time rise_time {{0.5 * clock_period}} clock_period)

.tran 1p {{3 * clock_period}}

.control
run
wrdata {output_file} C R D nand0 nand1 nand2 nand3 Q nand5
quit
.endc
.end
"""
        
        corners = ['mos_ss_mismatch']
        runs_per_corner = {'mos_ss_mismatch': 50}
        setups = np.arange(-110e-12, -39e-12, 1e-12)
        
        try:
            data_setup = self.load_data('setup2.pickle')
            print("  ℹ Loaded cached simulation data")
        except FileNotFoundError:
            print("  ℹ Running setup simulations...")
            data_setup = [{} for _ in range(setups.size)]
            for n, setup in enumerate(setups):
                if self.verbose and n % 10 == 0:
                    print(f"    Setup time {setup*1e12:.0f}ps ({n}/{len(setups)})...")
                for corner in corners:
                    def sim(vdd=self.vdd, temperature=self.temperature, rising_edge=True):
                        data0 = 0 if rising_edge else vdd
                        data1 = vdd if rising_edge else 0
                        return self.run_sim(template, mos_corner=corner,
                                          vdd=vdd, temperature=temperature,
                                          data0=data0, data1=data1,
                                          data_change=setup) | {'vdd': vdd, 'rising_edge': rising_edge}
                    
                    if corner in runs_per_corner:
                        for run in range(runs_per_corner[corner]):
                            for rising_edge in [True, False]:
                                data_setup[n][f'{corner}_{run:04}_{int(rising_edge)}'] = sim(
                                    vdd=self.vdd_model(), temperature=self.temperature_model(), rising_edge=rising_edge)
                    else:
                        for rising_edge in [True, False]:
                            data_setup[n][f'{corner}_{int(rising_edge)}'] = sim(rising_edge=rising_edge)
            self.save_data('setup2.pickle', data_setup)
        
        self.data_cache['setup'] = data_setup
        
        # Calculate setup costs
        print("\n  Calculating setup cost metric...")
        setup_costs = np.array([max(self.setup_cost(v, setup) for v in d.values()) 
                               for setup, d in zip(setups, data_setup)])
        
        # Find optimal setup (minimum cost)
        optimal_idx = np.argmin(setup_costs)
        optimal_setup = setups[optimal_idx]
        optimal_cost = setup_costs[optimal_idx]
        
        print(f"\n  Optimal setup time: {optimal_setup*1e12:.1f} ps")
        print(f"  Corresponding cost: {optimal_cost*1e12:.1f} ps")
        
        # Self-checking
        setup_error = abs(optimal_setup - self.specs.setup_time_target)
        setup_pass = setup_error <= self.specs.setup_time_tolerance
        
        self.results.append(TestResult(
            "Setup Time",
            TestStatus.PASS if setup_pass else TestStatus.FAIL,
            optimal_setup,
            self.specs.setup_time_target,
            message=f"Tolerance: ±{self.specs.setup_time_tolerance*1e12:.1f}ps"
        ))
        
        # Generate plots
        if self.save_plots:
            self.plot_setup_cost(setups, setup_costs, optimal_setup)
        
        return setup_pass
    
    # ==================== TEST 3: HOLD TIME ====================
    
    def test_hold_time(self) -> bool:
        """Test hold time with self-checking"""
        print("\n" + "="*70)
        print("TEST 3: HOLD TIME CHARACTERIZATION")
        print("="*70)
        
        template1 = """
.temp {temperature}
.lib cornerMOSlv.lib {mos_corner}
.include ../magic/fdc_dense.spice

Vdd VDD GND {vdd}
Vss VSS GND 0
.param rise_time=0p
.param clock_period=1n
Vresetn R GND {vdd}
Vdata D GND PULSE({data0} {data1} {{2 * clock_period + {data_change}}} 0 0 0)
Vclk C GND PULSE(0 {vdd} clock_period rise_time rise_time {{0.5 * clock_period}} clock_period)

.tran 1p {{3 * clock_period}}

.control
run
wrdata {output_file} C R D nand0 nand1 nand2 nand3 Q nand5
quit
.endc
.end
"""
        
        template2 = """
.temp {temperature}
.lib cornerMOSlv.lib {mos_corner}
.include ../magic/fdc_dense.spice

Vdd VDD GND {vdd}
Vss VSS GND 0
.param rise_time=0p
.param clock_period=1n
Vresetn R GND {vdd}
Vdata D GND PULSE({data1} {data0} {{1.5 * clock_period}} 0 0 {{0.5 * clock_period + {data_change}}} 0)
Vclk C GND PULSE(0 {vdd} clock_period rise_time rise_time {{0.5 * clock_period}} clock_period)

.tran 1p {{3 * clock_period}}

.control
run
wrdata {output_file} C R D nand0 nand1 nand2 nand3 Q nand5
quit
.endc
.end
"""
        
        corners = ['mos_ff_mismatch']
        runs_per_corner = {'mos_ff_mismatch': 250}
        holds = np.arange(5e-12, 11e-12, 1e-12)
        
        try:
            data_hold = self.load_data('hold5.pickle')
            print("  ℹ Loaded cached simulation data")
        except FileNotFoundError:
            print("  ℹ Running hold simulations (4 patterns × 250 runs × 6 hold times)...")
            data_hold = [{} for _ in range(holds.size)]
            for n, hold in enumerate(holds):
                print(f"    Hold time {hold*1e12:.0f}ps ({n+1}/{len(holds)})...")
                for corner in corners:
                    def sim(vdd=self.vdd, temperature=self.temperature, rising_edge=True, template_num=1):
                        data0 = 0 if rising_edge else vdd
                        data1 = vdd if rising_edge else 0
                        template = template1 if template_num == 1 else template2
                        return self.run_sim(template, mos_corner=corner,
                                          vdd=vdd, temperature=temperature,
                                          data0=data0, data1=data1,
                                          data_change=hold) | {'vdd': vdd, 'rising_edge': rising_edge, 'template': template_num}
                    
                    if corner in runs_per_corner:
                        for run in range(runs_per_corner[corner]):
                            for rising_edge in [True, False]:
                                for template_num in [1, 2]:
                                    data_hold[n][f'{corner}_{run:04}_{int(rising_edge)}_{template_num}'] = sim(
                                        vdd=self.vdd_model(), temperature=self.temperature_model(), 
                                        rising_edge=rising_edge, template_num=template_num)
                    else:
                        for rising_edge in [True, False]:
                            for template_num in [1, 2]:
                                data_hold[n][f'{corner}_{int(rising_edge)}_{template_num}'] = sim(
                                    rising_edge=rising_edge, template_num=template_num)
            self.save_data('hold5.pickle', data_hold)
        
        self.data_cache['hold'] = data_hold
        
        # Check which hold times have violations
        print("\n  Checking hold violations...")
        violates_hold = np.array([not all(self.meets_hold(trace) for trace in d.values()) for d in data_hold])
        
        # Find minimum hold time where no violations occur
        if np.any(violates_hold):
            hold_time = holds[np.where(violates_hold)[0][-1] + 1]
        else:
            hold_time = holds[0]
        
        print(f"\n  Measured hold time: {hold_time*1e12:.1f} ps")
        print(f"  Violations at: {holds[violates_hold]*1e12} ps")
        
        # Self-checking
        hold_error = abs(hold_time - self.specs.hold_time_target)
        hold_pass = hold_error <= self.specs.hold_time_tolerance
        
        self.results.append(TestResult(
            "Hold Time",
            TestStatus.PASS if hold_pass else TestStatus.FAIL,
            hold_time,
            self.specs.hold_time_target,
            message=f"Tolerance: ±{self.specs.hold_time_tolerance*1e12:.1f}ps"
        ))
        
        return hold_pass
    
    # ==================== PLOTTING FUNCTIONS ====================
    
    def plot_output_delay(self, data: Dict):
        """Generate output delay plot"""
        def corner_color(corner):
            if corner == 'mos_ff': return 'red'
            if corner == 'mos_ss': return 'blue'
            if corner == 'mos_tt': return 'green'
            if corner in ['mos_sf', 'mos_fs']: return 'black'
            return 'grey'
        
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(10, 8))
        c0 = 'mos_tt'
        for j, sig in enumerate(['C', 'R', 'D']):
            axs[j].plot(data[c0]['t'][sig] * 1e9, data[c0]['x'][sig], color=f'C{j}')
            axs[j].set_ylabel(sig, rotation=0)
        for corner in data:
            axs[-1].plot(data[corner]['t']['Q'] * 1e9, data[corner]['x']['Q'],
                        color=corner_color(corner), linewidth=1, alpha=0.7)
        axs[-1].set_ylabel('Q', rotation=0)
        for ax in axs:
            ax.grid()
            ax.set_yticks([0, 1.2], ['0 V', '1.2 V'])
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
        axs[-1].set_xlabel('Time (ns)')
        plt.subplots_adjust(hspace=0)
        plt.suptitle('Output Delay Simulation (All Corners)', y=0.95)
        plt.savefig('output_delay_plot.png', dpi=150, bbox_inches='tight')
        print("  ℹ Saved plot: output_delay_plot.png")
        plt.close()
    
    def plot_setup_cost(self, setups: np.ndarray, costs: np.ndarray, optimal: float):
        """Generate setup cost plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(setups * 1e12, costs * 1e12, 'b-', linewidth=2)
        plt.axvline(x=optimal * 1e12, color='r', linestyle='--', label=f'Optimal: {optimal*1e12:.1f}ps')
        plt.axvline(x=self.specs.setup_time_target * 1e12, color='g', linestyle=':', 
                   label=f'Target: {self.specs.setup_time_target*1e12:.1f}ps')
        plt.grid()
        plt.xlabel('Setup Time (ps)')
        plt.ylabel('Cost (Output Delay - Setup) (ps)')
        plt.title('Setup Cost Optimization')
        plt.legend()
        plt.savefig('setup_cost_plot.png', dpi=150, bbox_inches='tight')
        print("  ℹ Saved plot: setup_cost_plot.png")
        plt.close()
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def load_data(self, filename: str) -> any:
        """Load pickled data"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def save_data(self, filename: str, data: any):
        """Save data to pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def generate_report(self) -> bool:
        """Generate final test report"""
        print("\n" + "="*70)
        print("FINAL TEST REPORT")
        print("="*70)
        
        for result in self.results:
            print(f"  {result}")
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        total = len(self.results)
        
        print("\n" + "-"*70)
        print(f"  Total Tests: {total}")
        print(f"  Passed: {passed} ✓")
        print(f"  Failed: {failed} ✗")
        print(f"  Pass Rate: {100*passed/total:.1f}%")
        print("="*70)
        
        return failed == 0


def main():
    """Main test execution"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " AUTOMATED SELF-CHECKING FLIP-FLOP TIMING TESTBENCH ".center(68) + "║")
    print("║" + " Based on daniestevez/flip-flop-timing ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Initialize testbench with specifications
    specs = TimingSpecs()
    tb = FlipFlopAutomatedTestbench(specs, save_plots=True, verbose=True)
    
    # Run complete test suite
    try:
        test1_pass = tb.test_output_delay()
        test2_pass = tb.test_setup_time()
        test3_pass = tb.test_hold_time()
        
        # Generate final report
        all_passed = tb.generate_report()
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Flip-flop meets timing specifications.")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED! Review results above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 ERROR: Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())