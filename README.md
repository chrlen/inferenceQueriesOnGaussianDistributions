# Inference queries on gaussian distributions

The code for my bachelors thesis. The goal was to evaluate the performance of the operations of conditioning and marginalization comparing two parameterizations of gaussian models in terms of speed.

To run change directory to "scripts" and choose one of the following:

- scripts/runAll.sh runs generation, timing and plotting
- scripts/generateModels.sh generates models
- scripts/timeOnly.sh runs all timing operations on the generated models
- scripts/plotOnly.sh generates plots from the data gathered in the timing operations
- scripts/cleanModels.sh deletes all pngs and csvs in the intel directory and  all csvs in the models directory
- scripts/clean.sh delets all plots and csv files in the intel directory and  all csv files in the models directory

The following files contain the main elements:

- *distribution.py* holds an implementation of the multivariate gaussian distribution capable of performing marginalization and conditioning as well as being able to convert between the canonical and the mean form. The covariance/information matrix is represented in a compressed sparse column format.

- *distributionDense.py* is similiar to distribution.py in terms of functionality, but represents the covariance/information matrix in a two dimensional array.

- *generateSparseModels.py* draws random sparse symmetric matrices with differents sizes and densities

- All files beginning with "time" take measurements of the Operations of objects of the distribution classes or do evaluate the performance of atomic operations, e.g. set operations, matrix-vector multiplication or matrix inversion
