#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# download Path Nirvana Sinhala dataset
#wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# extract
mkdir pn_dataset
tar -xjf pn_dataset.tar.bz2 --directory pn_dataset
# create train-val splits
shuf pn_dataset/metadata.csv > pn_dataset/metadata_shuf.csv
head -n 3000 pn_dataset/metadata_shuf.csv > pn_dataset/metadata_train.csv
tail -n 300 pn_dataset/metadata_shuf.csv > pn_dataset/metadata_val.csv
mv pn_dataset $RUN_DIR/recipes/pathnirvana/
rm pn_dataset.tar.bz2