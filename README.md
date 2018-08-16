# duplicate-question-identification
A Siamese GRU Network to identify duplicate question pairs provided by Quora.com

## Description
The model takes two questions as inputs, and outputs a label predicting whether the two input sentences are semantic equivalent, or can be answered with the exact same answer. The model is consisted of two subnetworks: a Siamese GRU layer that encodes each question into a vector, and a fully connected layer that measures the semantic similarity of the two encoded vectors. The training data is provided by Quora.com, which includes 404,351 question pairs labeled either "duplicate" or "non-duplicate". The training/validation/test split is the same as proposed by [Wang et al](https://arxiv.org/pdf/1702.03814.pdf). A final test accuracy of 87.1% is achieved with a Bi-Directional GRU layer with the adapted attention mechanism.
For additional details, refer to the [final report](https://github.com/frankyanghkust/duplicate-question-identification/blob/master/report.pdf).
