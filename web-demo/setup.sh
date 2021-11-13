#!/bin/bash

export OUTPUT_DIR=./data

mkdir -p $OUTPUT_DIR/generated_outputs/queue
mkdir -p $OUTPUT_DIR/generated_outputs/inputs
mkdir -p $OUTPUT_DIR/generated_outputs/final

touch $OUTPUT_DIR/generated_outputs/queue/queue.txt
touch $OUTPUT_DIR/generated_outputs/inputs/tmp.txt
touch $OUTPUT_DIR/generated_outputs/final/tmp.txt

cd strap-frontend
npm install
cd ..
