#!/bin/bash
BASEDIR=$(realpath $(dirname "$0"))
PARENTDIR=$(dirname "$BASEDIR")
DATADIR=$PARENTDIR/data # original data directory
TMPDIR=$PARENTDIR/tmp   # move original files to this directory
TMPDIR1=$PARENTDIR/tmp1 # temporary directory to store splitted files

# create temporary directories
mkdir -p $TMPDIR
mkdir -p $TMPDIR1

# split file from data directory into tmp directory
for file in $DATADIR/*.txt; do
    filename=$(basename "$file")
    filesize=$(du -m "$DATADIR/$filename" | cut -f1)
    # split file if file size is greater than 550 mb
    if [ $filesize -ge 550 ]; then
        filename="${filename%.*}" # remove extension
        echo "Splitting $filename"
        # split file into 500 mb files in tmp1 directory
        split -b 500M -d --numeric-suffixes=1 --additional-suffix=.txt $DATADIR/$filename.txt $TMPDIR1/$filename-
        # move original file to tmp directory
        mv $DATADIR/$filename.txt $TMPDIR
    fi
done

if [ -f $TMPDIR1/* ]; then
    mv $TMPDIR1/* $DATADIR  # move back contents from tmp1 directory to data directory
fi

if [ -d $TMPDIR1 ]; then
    rm -rf $TMPDIR1 # remove tmp1 directory
fi
