#!/bin/bash

# Makes programs, downloads sample data, trains a glove model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

CORPUS=${10}
VOCAB_FILE=$1
VOCAB_MIN_COUNT=5
WINDOW_SIZE=$9
COOCCURRENCE_FILE=$2
ARRAY=(${COOCCURRENCE_FILE//./ })
COOCCURRENCE_SHUF_FILE="${ARRAY[0]}.shuf.${ARRAY[1]}"
BUILDDIR=build
SAVE_FILE=$7
VERBOSE=2
MEMORY=4.0
VECTOR_SIZE=$3
MAX_ITER=$4
BINARY=2
NUM_THREADS=${11}
X_MAX=$5
LEARNING_RATE=$6
DIR=$GLOVE #"$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
TMP_DIR=$8

cd $DIR; make

$DIR/$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $TMP_DIR/$VOCAB_FILE
$DIR/$BUILDDIR/cooccur -memory $MEMORY -vocab-file $TMP_DIR/$VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $TMP_DIR/$COOCCURRENCE_FILE
$DIR/$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $TMP_DIR/$COOCCURRENCE_FILE > $TMP_DIR/$COOCCURRENCE_SHUF_FILE
if [[ $? -eq 0 ]]
then
	$DIR/$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $TMP_DIR/$COOCCURRENCE_SHUF_FILE -eta $LEARNING_RATE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $TMP_DIR/$VOCAB_FILE -verbose $VERBOSE
fi

rm "$TMP_DIR/$VOCAB_FILE"
rm "$TMP_DIR/$COOCCURRENCE_FILE"
rm "$TMP_DIR/$COOCCURRENCE_SHUF_FILE"

