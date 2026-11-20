#!/usr/bin/env bash
set -euo pipefail

#logs/setup01_run_20261115_05*

find Dockerfile ./*.sh config/ doc/  plan/20261115_01_review/ -type f \
    | grep -v 'config6' \
    |grep -v ".png$" \
    | sort \
    | while IFS= read -r f; do
      echo "//start of  ${f}"
      cat "${f}"
      #ls "${f}"
      echo ""
    done

cat <<EOF


Der AI Agent hat schwer verstaendliches walkthrough.md dokument in
plan/20261115_01_review geschrieben.

Schau dir den Quellcode und die Scripte an, mache ein Review und
schreibe ein ordentliches walkthrough dokument.

EOF
