#!/bin/bash
set -e

source setup-pdk

echo "=== Checking for OSDI models ==="
ls -la ${PDK_ROOT}/ihp-sg13g2/libs.tech/ngspice/osdi/ || echo "OSDI directory doesn't exist"

echo ""
echo "=== Searching for .va source files ==="
find ${PDK_ROOT} -name "*.va" -type f

echo ""
echo "=== Checking for OpenVAF compiler ==="
which openvaf || echo "OpenVAF not installed"