---
configs:
- config_name: raw
  data_files:
    - path: tourism.csv
      split: train
- config_name: train_features
  data_files:
    - path: Xtrain.csv
      split: train
- config_name: test_features
  data_files:
    - path: Xtest.csv
      split: test
- config_name: train_labels
  data_files:
    - path: ytrain.csv
      split: train
- config_name: test_labels
  data_files:
    - path: ytest.csv
      split: test
---
# Tourism Package Prediction Dataset
