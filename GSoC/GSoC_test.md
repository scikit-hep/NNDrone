# Google Summer of code test

In order to demonstrate some of the required skills for the project,
students will need to perform a small task designed to test basic python
knowledge, in addition to basic machine learning principles.

### Task

In the [letter](https://arxiv.org/abs/1712.09114) describing the initial demonstration of a drone,
simulated data was used to create a neural network with [SciKit-learn](http://scikit-learn.org/).
The simulated data contained a signal B decay and a background D decay,
using six features of the data.

Students are required to train a new classifier with SciKit-learn, using
the example [signal](signal_data.p) and [background](background_data.p) data provided.
However, students are limited to two features of their choice.

### Evaluation

Performance of the new classifier will be tested using 1 million signal and
1 million background data points that students do not have access to,
but that are created in an equivalent way to the training data.

The test score will be determined according to the following figure-of-merit
popular in particle physics:

$\sigma=\alpha\frac{S}{\sqrt{S+B}}$

where $S$ is the number of signal events passing the classifier in the
training sample and $B$ is the number of background events passing the classifier.

##### The factor $\alpha$

*** Warning **** The bonus multiplier is not for the faint-hearted.

The factor $\alpha$ is a bonus multiplier for those wishing an extra challenge.
This is defined as

![](http://latex.codecogs.com/svg.latex?%5Calpha%3D%281%2B%5Cfrac%7B1%7D%7B%5Cchi%5E%7B2%7D%7D)

$\alpha=(1+\frac{1}{\chi^{2})$

where $\chi^{2}$ is taken as the sum of point-by-point differences in
the output predictions between the drone and the SciKit-learn classifier.

# Submitting results

Those wishing to submit test results should create a new merge request
which contains a new folder in the GSoC directory (with the name of the folder
as the initials of the applicant).
This folder should contain a pickle file of the created SciKit-Learn classifier.
If the bonus multiplier is attempted, a drone network in json
format should also be included in the folder.
