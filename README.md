# Guided Proofreading

Automatic cell image segmentation methods in connectomics produce merge and
split errors, which require correction through proofreading.

Guided Proofreading includes two classifiers that automatically recommend candidate merges and splits to the proofreader. These classifiers use a convolutional neural network (CNN) that has been trained with
errors in automatic segmentations against expert-labeled ground truth. 

Our classifiers detect potentially-erroneous regions by considering a large context
region around a segmentation boundary. Corrections can then be performed by a
user with yes/no decisions,  which results in faster correction times than previous
proofreading methods. We also present a fully-automatic mode that uses a
probability threshold to make merge/split decisions.

# Usage

## How to check a pair of segments for a split error?

## How to check a segment for merge errors?

## How to run split and merge error correction automatically on a whole volume?

## How to use the user interface?
