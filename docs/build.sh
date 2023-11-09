#!/bin/sh

sh docs/clean.sh
make html
cd docs/build/html
python -m http.server 8000

