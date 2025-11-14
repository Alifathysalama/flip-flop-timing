#!/bin/bash
set -e

# Set up PDK environment
source setup-pdk

echo "PDK_ROOT: $PDK_ROOT"
echo "PDK: $PDK"
echo "SPICE_USERINIT_DIR: $SPICE_USERINIT_DIR"

echo ""
echo "=== Extracting SPICE netlist from Magic layout ==="
cd magic
magic -noconsole -dnull -rcfile ${PDK_ROOT}/ihp-sg13g2/libs.tech/magic/ihp-sg13g2.magicrc magic-extract.tcl
cd ..

echo ""
echo "=== Running Testbench ==="
cd testbench
python3 automated_testbench.py