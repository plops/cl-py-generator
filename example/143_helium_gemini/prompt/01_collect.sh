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
for i in "${FILES[@]}"; do
        echo "// start of $i"
        cat "$i"
done | xclip


for i in "${FILES[@]}"; do
        echo "// start of $i"
        ls "$i"
done


database_loading_guide.md describes how we can fix a problem in p04_host.py.
the current version takes a long time to start up. however, i don't want to modify the python code.
i want to fix the underlying s-expressions in gen04.lisp that are transpiled to python code.

create a implementation plan for the fix. show the code changes that need to be implemented.
assume that the ai that performs the implementation is much less intelligent and informed than you.
so you have to split the task into individual steps and run format_lisp and generate_python after each one.

and in order to run the full python program for testing, the agent has to execute:

/home/kiel/stage/cl-py-generator/example/143_helium_gemini/source04/tsum/.venv/bin/python /home/kiel/stage/cl-py-generator/example/143_helium_gemini/source04/tsum/.venv/bin/uvicorn p04_host:app --port 5001

and check that the response contains properly formatted entries from the database with content similar to this:
```
https://www.youtube.com/watch?v=WFSgliBKpjM
ID: 13020 | Model: gemini-2.5-flash-preview-09-2025

Reviewing Body: The Council for Comparative Civilization Studies (CCCS)

Abstract:

This lecture provides a comparative analysis of the enduring legal, philosophical, and cultural contributions of ancient Hebrew civilization (Judaism/Israelites) to contemporary Western society. The presentation begins by
```
