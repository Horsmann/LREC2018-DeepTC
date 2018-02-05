# LREC2018-DeepTC

Example projects that are used in the publication:

Tobias Horsmann and Torsten Zesch (2018). DeepTC - An Extension of DKPro Text Classification for Fostering Reproducibility of Deep Learning Experiments. In proceedings of International Conference on Language Resources and Evaluation (LREC). Miyazaki, Japan.

*Provided Examples*
The example projects demonstrate the usage of DKPro TC for a sequence classification task (part-of-speech tagging) for the deep learning frameworks Keras, DyNet and Deeplearning4j. For a direct comparison to a shallow learning framework a comparable setup with a shallow Conditional Random Field is implemented.

The Deeplearning4j and CRF example are runnable as-is as both are Java-based projects. DyNet and Keras requires that the respective framework is already locally installed on the user's computer.

*Results*
The results are written by default to the user's home directory to the "/Desktop" folder `org.dkpro.lab`. In this folder, a number of sub-folder is created for each execution of an experiment. The folder that are named after the machine learning framework, e.g. Deeplearning4j, contain the results in a file `results.txt`
