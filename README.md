# Causal-Ex: Causal Graph-based Micro and Macro Expression Spotting

Detecting concealed emotions within apparently normal expressions is crucial for identifying potential mental health issues and facilitating timely support and intervention. The task of spotting macro and micro-expressions involves predicting the emotional timeline within a video, accomplished by identifying the onset, apex, and offset frames of the displayed emotions. Utilizing foundational facial muscle movement cues, known as facial action units, boosts the accuracy. However, an overlooked challenge from previous research lies in the inadvertent integration of biases into the training model. These biases arising from datasets can spuriously link certain action unit movements to particular emotion classes. We tackle this issue by novel replacement of action unit adjacency information with the action unit causal graphs. This approach aims to identify and eliminate undesired spurious connections, retaining only unbiased information for classification. 

(Will update full code soon...) 

### Main Contribution

- **Causal AU Relation Graph**:  
  Generate AU-based causal relation graphs from a provided video using the causal structure discovery algorithm (FCI).
  
- **Improved Architecture**:  
  Introduce an enhanced framework for macro- and micro-expression spotting.

### Implementation Files

- **model.py**:  
  The architecture of Causal-Ex with the causal structure learning algorithm implemented.
  
- **train.py**:  
  Explains how the results proposal was computed.
  
- **train_utils.py**:  
  Separates all parameters into those that will and won’t experience regularizing weight decay.

### Acknowledgements

- This project was inspired by the [AUW-GCN](https://github.com/xjtupanda/auw-gcn).
