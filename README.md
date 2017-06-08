# Ant Colony Optimization with Linear Discriminant Analysis

This repo is used to perform Ant Colony optimization for feature reduction in a dataset. Uses Linear Discriminant Analysis as the fitness function.
The research paper referred and the implemented algorithm can be found here:
[link](https://github.com/Nikhil12321/major2/blob/master/Docs/Untitled.pdf)

## How to start?
1. The file you would want to run is p1.py which contains the entire algorithm.
2. Near the end of the file under declarations, you will see a variable 'filename' which contains the name of the .csv file you want to use as your data.
3. All other declarations are various variables such as number of ants, number of iterations, and others which you can tweak to your interest.
4. You can also change the fitness change to suit your interest. We have used LDA, a statistical classifier.

## Anything else that is needed?
YES!!
1. **A major** part of the algorithm is calculating mutual information between two features, a class and a features and a class and two features whose variables are **mi_fc, mi_ff, cmi_ffc**. To calculate these, we have used a library [gcmi](https://github.com/robince/gcmi). Thanks to them!
2. The gcmi library **only works for continuous** variables and may fail for sparsely populated datasets.

## Anything for optimization?
1. Calculation of mutual information takes time and this can cause slowdown of program in each iteration. Since the mutual information variables are data structures, you can save them using pickle once and load for further runs.


## What can I contribute?
1. Since this program depends upon external library for calculation of mutual information, we can make it independent by inbuilting it.
2. The program only works for features that are continuous values. We can make it generic by including support for dicrete features as well.


**Thanks**
gcmi.py is used. Thanks to the wonderful work of -
RAA Ince, BL Giordano, C Kayser, GA Rousselet, J Gross and PG Schyns
"A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula"
bioRxiv doi:10.1101/043745

Update pheromone trails
run p1.py
