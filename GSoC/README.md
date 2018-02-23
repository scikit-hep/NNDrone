# Google Summer of code test

## Extension and characterisation of the LHCb data isolation drone neural networks

This repository contains an exercise to evaluate those students interested in applying for
the *Extension and characterisation of the LHCb data isolation drone neural networks project*,
included in the Google Summer of Code (GSoC) program and offered by members of the LHCb Collaborations at CERN.
The detailed description of the project can be found [here](http://hepsoftwarefoundation.org/gsoc/2018/proposal_LHCbHEPDrone.html).

## Exercise for Candidate Students

In order to demonstrate some of the required skills for the project,
students will need to perform a small task designed to test basic python
knowledge, in addition to basic machine learning principles.
The overall task is split in two smaller tasks, with one extra more advanced exmple,
which is optional, but illustrates more widely the project in HEP ecosystem.


### Task 1: *Reproduce available model*

In the [letter](https://arxiv.org/abs/1712.09114) describing the initial demonstration of a drone,
simulated data was used to create a neural network with [SciKit-learn](http://scikit-learn.org/).
The simulated data contained a signal B decay and a background D decay,
using six features of the data.

As a start, students are asked to reproduce the results of the paper using already available
script in [test_model](test_model.py).

In preparation for this task:

  * Install `python2`
  * Install project requirements - `numpy`, `sklearn`, `scipy`, `matplotlib`
  * Clone this repository to your local machine
  * Set up the environment by running: `cd /path/to/HEPDrone/; ./setup.sh`

Once you have set up your environment with all the software listed above,
you should be able to run `cd /path/to/HEPDrone/GSoC; ./test_model.py <YOUR SURNAME>`.
This should run very quickly and produce some plots in `GSoC/plots`.
The deliverable of this task is precisely that folder and the plots in it.
After being executed and saved, you can add this to the rest of the required
as described in [section Submitting results](#submitting-results).

### Task 2: *Train a drone neural net*

Task 2 is a bit more complicated and requires some understanding of the
framework and a little bit of coding.

Students are required to train a new classifier with SciKit-learn, using
the example [signal](../data/signal_data.p) and [background](../data/background_data.p) data provided.
However, students are limited to two features of their choice. Trying and
comparing different feature choice is encouraged, but not required.

Performance of the new classifier will be tested using 1 million signal and
1 million background data points that students do not have access to,
but that are created in an equivalent way to the training data.

The test score will be determined according to the following figure-of-merit
popular in particle physics:

![](http://latex.codecogs.com/svg.latex?%5Csigma%3D%5Calpha%5Cfrac%7BS%7D%7B%5Csqrt%7BS%2BB%7D%7D)

where S is the number of signal events passing the classifier in the
training sample and B is the number of background events passing the classifier.

##### The factor ![](http://latex.codecogs.com/svg.latex?%5Calpha)

*** Warning **** The bonus multiplier is not for the faint-hearted.

The factor ![](http://latex.codecogs.com/svg.latex?%5Calpha) is a bonus multiplier for those wishing an extra challenge.
This is defined as

![](http://latex.codecogs.com/svg.latex?%5Calpha%3D%281%2B%5Cfrac%7B1%7D%7B%5Cchi%5E%7B2%7D%7D%29)

where ![](http://latex.codecogs.com/svg.latex?%5Cchi%5E%7B2%7D) is taken as the sum of point-by-point differences in
the output predictions between the drone and the SciKit-learn classifier.

### Task 3 (Optional, Advanced): *From start to finish*

This task is even more involved that the previous two. It requires more
deep understanding and skills in the wider data analysis field.

In preparation for this task:

  * Install [ROOT v6.12/06](https://root.cern.ch) framework (Download from [here](https://root.cern.ch/releases))
    - Packages exist for most Operating Systems
    - Manual build instructions can be found [here](https://root.cern.ch/building-root)
    - Example build script:
      ```bash
        cd /path/to/ROOT/source
        mkdir build
        cd build
        CFLAGS="${CFLAGS} -pthread" \
          CXXFLAGS="${CXXFLAGS} -pthread" \
          LDFLAGS="${LDFLAGS} -pthread -Wl,--no-undefined" \
          cmake -C settings.cmake ../
        make -j<num_threads>
        make DESTDIR="/usr or /usr/local or /path/to local folder" install
      ```
      example settings.cmake can be found [here](https://pastebin.com/jADQQr40)
  * Clone [RapidSim](https://github.com/gcowan/RapidSim) project
  * Follow the instructions and build the `RapidSim` project

The task is to generate a new signal data, create a new model and evaluate its performance. To summarize the steps:

  * Use the [signal decay](Bd2JpsiK.decay) and [signal config](Bd2JpsiK.config) files to generate new signal events with `RapidSim`
  * Use the newly generated `*.root` file and this [template script](../scripts/gen_train.py), which you need to adapt, to make a new `signal_data.p` sample for training
  * Use same background data as in [Task 2](#task-2-train-a-drone-neural-net)
  * Use **SciKit-learn MLP** to train a new model ([example script](../skLearn-classifiers/train-skLearn.py))
  * Follow the steps from the previous two tasks to create a drone neural net and evaluate its performance

As a submission, we would accept:

  * Explanation of the steps followed (Markdown document is sufficient, but any doc format would do)
  * The `*.root` files used to generate the training samples
  * The saved `MLP` and `drone model` in `*.pkl` files
  * The new plots showing the performance of the `MLP` and `drone network`

### Evaluation


# Submitting results

Those wishing to submit test results should create a new merge request
which contains a new folder in the GSoC directory (with the name of the folder
as the initials of the applicant).
This folder should contain a pickle file of the created SciKit-Learn classifier.
If the bonus multiplier is attempted, a drone network in JSON
format should also be included in the folder.

Once you complete any of the tasks of this exercise, please send us by e-mail the requested deliverables and the answers to the proposed questions at: s.benson@cern.ch, k.gizdov@cern.ch
