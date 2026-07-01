#!/bin/bash
cp ../../../dotemacs/.emacs .
docker build -t antigravity-sandbox:26.04 .
rm -f .emacs
