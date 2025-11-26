#!/bin/bash
set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PDK paths to use the local IHP-Open-PDK directory
export PDK_ROOT="${SCRIPT_DIR}/IHP-Open-PDK"
export PDK=ihp-sg13g2
export SPICE_USERINIT_DIR="${PDK_ROOT}/${PDK}/libs.tech/ngspice"

# Verify the PDK directory exists
if [ ! -d "${PDK_ROOT}" ]; then
    echo "ERROR: PDK directory not found at ${PDK_ROOT}"
    echo "Please ensure IHP-Open-PDK submodule is initialized:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "=== PDK Configuration ==="
echo "PDK_ROOT: ${PDK_ROOT}"
echo "PDK: ${PDK}"
echo "SPICE_USERINIT_DIR: ${SPICE_USERINIT_DIR}"
echo ""

echo "=== Compiling OSDI models from Verilog-A sources ==="

# Create the OSDI directory if it doesn't exist
OSDI_DIR="${PDK_ROOT}/${PDK}/libs.tech/ngspice/osdi"
mkdir -p "${OSDI_DIR}"

# Compile each Verilog-A model to OSDI
VA_DIR="${PDK_ROOT}/${PDK}/libs.tech/verilog-a"

echo "Compiling psp103_nqs.osdi..."
openvaf "${VA_DIR}/psp103/psp103_nqs.va" -o "${OSDI_DIR}/psp103_nqs.osdi"

echo "Compiling r3_cmc.osdi..."
openvaf "${VA_DIR}/r3_cmc/r3_cmc.va" -o "${OSDI_DIR}/r3_cmc.osdi"

echo "Compiling mosvar.osdi..."
openvaf "${VA_DIR}/mosvar/mosvar.va" -o "${OSDI_DIR}/mosvar.osdi"

echo ""
echo "=== Verifying OSDI files ==="
ls -lh "${OSDI_DIR}/"

echo ""
echo "=== Testing ngspice availability ==="
which ngspice
ngspice --version

# works for the si-time bouquet, but not for ihp pdk
# echo ""
# echo "=== Extracting SPICE netlist from Magic layout ==="
# cd "${SCRIPT_DIR}/magic"
# magic -noconsole -dnull -rcfile "${PDK_ROOT}/${PDK}/libs.tech/magic/${PDK}.magicrc" magic-extract.tcl
# cd "${SCRIPT_DIR}"

# echo ""
# echo "=== Verifying extracted files ==="
# ls -lh "${SCRIPT_DIR}/magic/fdc_dense.spice"

# echo ""
# echo "First 10 lines of fdc_dense.spice:"
# head -10 "${SCRIPT_DIR}/magic/fdc_dense.spice"

echo ""
echo "=== Using precompiled SPICE netlist ==="

# Ensure magic folder exists
mkdir -p "${SCRIPT_DIR}/magic"

# Expected location for the SPICE netlist (used by testbench)
DEST_SPICE="${SCRIPT_DIR}/magic/fdc_dense.spice"

# Check if the SPICE file already exists at the expected location
if [ -f "${DEST_SPICE}" ]; then
    echo "SPICE netlist already present at: ${DEST_SPICE}"
    echo "Verifying file:"
    ls -lh "${DEST_SPICE}"
    echo ""
    echo "First 10 lines:"
    head -10 "${DEST_SPICE}"
else
    echo "ERROR: SPICE netlist not found at expected location:"
    echo "  ${DEST_SPICE}"
    echo "Please ensure fdc_dense.spice exists in the magic/ directory"
    exit 1
fi

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
cd "${SCRIPT_DIR}/testbench"
sed -i "s|> /dev/null 2>&1||g" automated_testbench.py
python3 automated_testbench.py
git checkout automated_testbench.py 2>/dev/null || true

echo ""
echo "=== Test Complete ==="