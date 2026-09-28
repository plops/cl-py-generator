#!/usr/bin/env bash
# download_ground_truth.sh — Lädt einen gelabelten Gesichts-Datensatz (LFW,
# Labeled Faces in the Wild) für die End-to-End-Validierung der Re-ID-Pipeline.
#
# LFW: >13.000 Web-Fotos bekannter Personen mit Identitäts-Labels, der
# Standard-Benchmark für Gesichtsverifikation. Wir nutzen die sklearn-Variante
# (fetch_lfw_people), die den Datensatz nach ~/scikit_learn_data cached.
#
# Nutzung:
#   ./download_ground_truth.sh                 # Default: min_faces_per_person=20
#   ./download_ground_truth.sh 30              # nur Personen mit >=30 Bildern
#
# Voraussetzung: das venv (.venv) mit scikit-learn. Fehlt es, siehe README.

set -euo pipefail
cd "$(dirname "$0")"

MIN_FACES="${1:-20}"

if [ ! -d .venv ]; then
    echo "Kein .venv gefunden. Erst einrichten:"
    echo "  uv venv && uv pip install -r requirements.txt"
    echo "  uv pip install onnxruntime opencv-python-headless"
    exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python - "$MIN_FACES" <<'PY'
import sys
from sklearn.datasets import fetch_lfw_people

min_faces = int(sys.argv[1])
print("Lade LFW (min_faces_per_person=%d) ..." % min_faces)
lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True,
                       resize=1.0, funneled=True)
n_persons = len(set(lfw.target))
print("Fertig: %d Bilder, %d Personen (gecacht in ~/scikit_learn_data)."
      % (len(lfw.images), n_persons))
print("Weiter mit:  ./collect.sh  bzw.  python face_dataset_validate.py --models <DIR>")
PY
