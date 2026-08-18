#!/bin/sh
# Build paper/paper.pdf with tectonic, then copy to docs/ for GitHub Pages.
set -eu
cd "$(dirname "$0")"
tectonic -X compile paper.tex --outdir .
cp -f paper.pdf ../docs/paper.pdf
echo "wrote paper/paper.pdf and docs/paper.pdf"
