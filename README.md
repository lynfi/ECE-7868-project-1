# ECE-7868-projects

## Language: 
Matlab

## Project 1

Use the PCA algorithm and Nearest Neighbor Method to classify objects in CIFAR-10 and ETH-80 and people’s identities in the AR face database. A detailed description of the method and the result can be found in the report - "ECE_7868_Project_1_Report.pdf"

## Project 2

Use PCA, ICA, and LDA to learnthe subspace of specific objects and compress the images. A detailed report is in "ECE_7868_Project_2_Report.pdf".

### Implementation

Dateset:

```Need to put the AR face dataset in the same directory. ```

File description:

```
Compression.m: main file of image compression 

Sample.m: main file of generating samples

LDA.m: the function of LDA

FastICA_25: (file folder) the function of ICA

mgs.m: the function of Gram-Smith Orthogonalization 

ECE_7868_Project_2_Report.pdf: report
```

Need to implement Compression.m and Sample.m

## Project 3

Use Gaussian Mixture Model to classify images in theri PCA, ICA and LDA subspace. A detailed report is in "ECE_7868_Project_3_Report.pdf"

### Implementation

Dateset:

```Need to put the AR face dataset ETH-80 and CIFAR-10 in the same directory. ```

File description:

```
AR: main file for AR dataset

ETH80: main file for ETH80 dataset 

CIFAR10: main file for CIFAR10 dataset 

LDA.m: the function of LDA

FastICA_25: (file folder) the function of ICA

mgs.m: the function of Gram-Smith Orthogonalization 

ECE_7868_Project_3_Report.pdf: report
```

Need to implement AR.m and ETH80.m and CIFAR10.m

WARNING: Sometimes the EM algorithm may converge to an ill-conditioned covariance matrix, which will cause an error. We need re-run the code in this case. (The probability is very small.)

### Reference

*Gross, Ralph, Jie Yang, and Alex Waibel. "Growing Gaussian mixture models for pose invariant face recognition." Pattern Recognition, 2000. Proceedings. 15th International Conference on. Vol. 1. IEEE, 2000.*
