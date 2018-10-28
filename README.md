# NNDrone: A collection of tools and algorithms to enable conversion of HEP ML to mass usage model.

Machine learning has proven to be an indispensable tool in the selection of interesting events in high energy physics.
Such technologies will become increasingly important as detector upgrades are introduced and data
rates increase by orders of magnitude. NNDrone is a toolkit to enable the creation of a drone
classifier from any machine learning classifier, such that different classifiers may be standardised
into a single form and executed in parallel. In the package, we demonstrate the capability of the drone neural
network to learn the required properties of the input neural network without the use of any training data,
only using appropriate questioning of the input neural network - https://arxiv.org/abs/1712.09114

## Original training

To demonstrate the toolkit:
1) We create simulated HEP data from the RapidSim package (https://arxiv.org/abs/1612.07489)
2) We train a neural network using SciKit-Learn
3) We train a drone to learn the features
4) We export this to the equivalent coding in the production environment using JSON
