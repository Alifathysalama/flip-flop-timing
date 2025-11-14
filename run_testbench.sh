#!/bin/bash
set -e

# Set up PDK environment
export PDK_ROOT=$(pwd)/IHP-Open-PDK
export PDK=ihp-sg13g2
export SPICE_USERINIT_DIR=$PDK_ROOT/$PDK/libs.tech/ngspice

# Show environment (for debugging)
echo "PDK_ROOT: $PDK_ROOT"
echo "PDK: $PDK"
echo "SPICE_USERINIT_DIR: $SPICE_USERINIT_DIR"

# Run testbench
cd testbench
python3 automated_testbench.py