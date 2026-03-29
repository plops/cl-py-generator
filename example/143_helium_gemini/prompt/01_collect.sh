cd /home/kiel/stage/cl-py-generator/example/143_helium_gemini

declare -a FILES=(
    ../../AGENTS.md
    source04/tsum/doc/database_loading_guide.md
    gen04.lisp
    source04/tsum/doc/architecture.md
    source04/tsum/p04_host.py
    ../../SUPPORTED_FORMS.md
    format_lisp.sh
    generate_python.sh
    
)

# Create output file in /dev/shm/
OUTPUT_FILE="/dev/shm/collected_files_$(date +%Y%m%d_%H%M%S).txt"

# Start with the prompt content
cat << 'EOF' > "$OUTPUT_FILE"

The current p04_host.py takes more than one minute to start up when processing a 1.6GB database file. This is unacceptable performance.

Important context: The Python code in p04_host.py is transpiled from gen04.lisp. We cannot modify the Python code directly because:
1. Changes would be lost when regenerating from Lisp
2. The Lisp code must remain valid and properly formatted
3. Parentheses errors in Lisp can break the entire transpilation process (see AGENTS.md)

The database_loading_guide.md contains detailed technical information about the loading process and specific optimization strategies.

Your task:
Create an implementation plan to fix the startup performance by modifying the Lisp code in gen04.lisp. Follow the recommendations in database_loading_guide.md.

Requirements:
- Split the work into individual steps suitable for a less experienced AI with not as much context.
- After each Lisp modification, run format_lisp.sh and generate_python.sh
- Ensure the Lisp code remains valid and can be formatted properly
- Verify the Python script works correctly after each change

Testing procedure:
Execute the following command to test the full program:
/home/kiel/stage/cl-py-generator/example/143_helium_gemini/source04/tsum/.venv/bin/python /home/kiel/stage/cl-py-generator/example/143_helium_gemini/source04/tsum/.venv/bin/uvicorn p04_host:app --port 5001

Verify the response contains properly formatted database entries like:
```
https://www.youtube.com/watch?v=WFSgliBKpjM
ID: 13020 | Model: gemini-2.5-flash-preview-09-2025

Reviewing Body: The Council for Comparative Civilization Studies (CCCS)

Abstract:

This lecture provides a comparative analysis of the enduring legal, philosophical, and cultural contributions of ancient Hebrew civilization (Judaism/Israelites) to contemporary Western society. The presentation begins by
```

EOF

# Track missing files
MISSING_FILES=()

for i in "${FILES[@]}"; do
    if [[ -f "$i" ]]; then
        echo "// start of $i" >> "$OUTPUT_FILE"
        cat "$i" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    else
        echo "WARNING: File not found: $i" >&2
        MISSING_FILES+=("$i")
        echo "// WARNING: File not found: $i" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
done

# Report missing files summary
if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo "WARNING: ${#MISSING_FILES[@]} file(s) were missing:" >&2
    for missing in "${MISSING_FILES[@]}"; do
        echo "  - $missing" >&2
    done
else
    echo "All files found successfully." >&2
fi

echo "Output written to: $OUTPUT_FILE" >&2

# Pipe to xclip for easy copying
cat "$OUTPUT_FILE" | xclip
