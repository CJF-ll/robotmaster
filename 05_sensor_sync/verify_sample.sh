#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build
cmake --build build -j
./build/sensor_sync config/sensor_sync.conf
diff -u data/expected_result.txt data/result.txt
echo "Public sample passed."
