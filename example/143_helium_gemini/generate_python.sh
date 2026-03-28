#!/bin/bash
# Generate p04_host.py from gen04.lisp

cd /home/kiel/stage/cl-py-generator/example/143_helium_gemini

echo "Generating p04_host.py from gen04.lisp..."
sbcl --non-interactive --load gen04.lisp

if [ $? -eq 0 ]; then
    echo "✓ p04_host.py generated successfully"
else
    echo "✗ Error generating p04_host.py"
    exit 1
fi
