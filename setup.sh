#!/bin/sh

git clone https://github.com/keon/algorithms.git
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
cd algorithms
# Use a separate branch to isolate Gemini commits.
git checkout -b gemini-unit-tests
pip install -r requirements.txt
