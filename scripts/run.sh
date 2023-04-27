#!/bin/bash

trap "exit" INT
set -e

rm -rf results
mkdir results
tmuxp load ./mysession.yaml

python merge_all_results.py