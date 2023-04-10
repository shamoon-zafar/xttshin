#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# download Path Nirvana Sinhala dataset
wget https://github.com/pnfo/pali-tts-dataset/releases/download/v1.1/pali_dataset.tar.bz2.partaa
wget https://github.com/pnfo/pali-tts-dataset/releases/download/v1.1/pali_dataset.tar.bz2.partab
# extract
mkdir pali_dataset
cat pali_dataset.tar.bz2.part* | tar -xjf --directory pali_dataset

shuf pali_dataset/metadata.csv > pali_dataset/metadata_shuf.csv

mv pali_dataset $RUN_DIR/recipes/pathnirvana/
rm pali_dataset.tar.bz2.part*