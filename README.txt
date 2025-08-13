# Causal-Ex: Bias-Aware Dynamic AU Causal Graph based Emotional Facial Expression Detection

Supplementary Materials 

## Proposed methods 

- Causal AU Relation Graph: generate AU-based causal relation graphs from a provided video, 
                            employing a causal structure discovery algorithm (FCI). 
- Improved Architecture: introduce an enhanced framework for macro and micro-expression spotting

## Usage of files

model.py: The architecture of the Causal-Ex with the causal structure learning algorithm implemented.
train.py: Explains how the results proposal was computed. 
train_utils.py: Separate out all parameters to those that will and won't experience regularizing weight decay.

## Acknowledgements

- This project was inspired by the AUW-GCN. (https://github.com/xjtupanda/auw-gcn)
