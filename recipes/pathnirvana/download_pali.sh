#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# download Path Nirvana Sinhala dataset
wget https://github.com/pnfo/pali-tts-dataset/releases/download/v1.0/pali_dataset.tar.bz2
# extract
mkdir pali_dataset
tar -xjf pali_dataset.tar.bz2 --directory pali_dataset

shuf pali_dataset/metadata.csv > pali_dataset/metadata_shuf.csv

mv pali_dataset $RUN_DIR/recipes/pathnirvana/
rm pali_dataset.tar.bz2