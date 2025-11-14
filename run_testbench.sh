#!/bin/bash
set -e

# Set up PDK environment
source setup-pdk

echo "PDK_ROOT: $PDK_ROOT"
echo "PDK: $PDK"
echo "SPICE_USERINIT_DIR: $SPICE_USERINIT_DIR"

echo ""
echo "=== Testing ngspice availability ==="
which ngspice
ngspice --version

echo ""
echo "=== Extracting SPICE netlist from Magic layout ==="
cd magic
magic -noconsole -dnull -rcfile ${PDK_ROOT}/ihp-sg13g2/libs.tech/magic/ihp-sg13g2.magicrc magic-extract.tcl
cd ..

echo ""
echo "=== Verifying files ==="
ls -lh magic/fdc_dense.spice
echo ""
echo "First 10 lines of fdc_dense.spice:"
head -10 magic/fdc_dense.spice

echo ""
echo "=== Testing ngspice with a simple circuit ==="
cat > /tmp/test.cir << 'EOF'
Simple resistor test
V1 1 0 DC 1
R1 1 0 1k
.control
run
print v(1)
quit
.endc
.end
EOF
ngspice /tmp/test.cir

echo ""
echo "=== Running Testbench ==="
cd testbench
sed -i "s|> /dev/null 2>&1||g" automated_testbench.py
python3 automated_testbench.py
git checkout automated_testbench.py 2>/dev/null || true