#!/bin/bash

# Path to the configuration file and the trained model
CONFIG_PATH="configs/deim_dfine/deim_hgnetv2_x_driveu.yml"
MODEL_PATH="trained/deim_dfine_hgnetv2_x_driveu_160e.pth"

# List of image paths (add your image paths here)
IMAGE_LIST=(
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/137219865055764.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/140088308099968.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/148832127252848.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/153841250014296.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/160135085960601.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/167433588523967.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/173065118047434.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/174569654452217.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/175696014435416.jpg"
    "/Users/yesung/Projects/DEIM/dataset/berlin_test/2ZlpPELm6gmnAO9Wa__NPQ/196030482351195.jpg"
    # Add more image paths as needed
)

# Iterate over each image in the list
for IMAGE_PATH in "${IMAGE_LIST[@]}"
do
    echo "Processing image: $IMAGE_PATH"
    
    # Run the Python inference command
    python tools/inference/torch_inf.py -c $CONFIG_PATH -r $MODEL_PATH --input $IMAGE_PATH
done

echo "Processing complete!"