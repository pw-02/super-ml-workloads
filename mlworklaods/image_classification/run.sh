#!/bin/bash

# Start the first Python script and run it in the background
python mlworklaods/image_classification/launch.py

# Start the second Python script and run it in the background
# python mlworklaods/image_classification/launch.py &

# Wait for all background processes to finish
wait
