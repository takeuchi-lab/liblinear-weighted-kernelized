# LIBLINEAR-Weights-Kernelized

Authors of this version: [Nagoya University, Department of Mechanical Systems Engineering, Machine Learning & Data Science Laboratory](https://www.mlds.mae.nagoya-u.ac.jp/) (Main author: [Hiroyuki Hanada](https://github.com/hana-hiro))

Original authors of LIBSVM: C. C. Chang and Chih-Jen Lin. (see the file `README.LIBSVM.original`)

Original authors of LIBLINEAR: R. E. Fan, K. W. Chang, C. J. Hsieh, X. R. Wang, and C. J. Lin. (see the file `README.LIBLINEAR.original`)

## tl;dr

-   Support vector machine (SVM) solvers [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) have two differences: one is the use of kernel functions (only LIBSVM accepts), the other is the regularization being imposed on the intercept term (only LIBLINEAR uses).
-   This library is made by modifying LIBLINEAR (introducing a part of LIBSVM codes) to train SVM with kernel functions and intercept regularization.
-   This implementation includes the source code of C++, and Python wrapper. (Other languages are not supported. **Although "matlab" folder exists, it is not supported.**)

## TODO

-   The strategy of caching kernel matrix is not well optimized. We would like to alter this in the future.

## Overview

For the support vector machine (SVM) and similar machine learning models, [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) are well-known solvers. However, the author needs slightly different formulations from these two libraries.

The requirements of SVM for the author's work (e.g, [Distributionally Robust Coreset Selection under Covariate Shift (published in TMLR)](https://openreview.net/forum?id=Eu7XMLJqsC)) are as follows:

1. specification of weights on training instances (that weight the loss function values),
2. use of kernel functions, and
3. the intercept term is regularized (see the appendix at the end)

Here, LIBSVM is fine with points 1 and 2, but not 3. On the other hand, LIBLINEAR is fine with points 1 and 3, but not 2. So the author modified LIBLINEAR so that it accepts kernel functions. More specifically, we introduced a part of LIBSVM implementations into LIBLINEAR codes.

### Caveats

-   Currently, only these models accept kernel functions. An error will be raised if a kernel function is specified for other models.
    -   L2 regularization + hinge loss (Command line option "-t 3"; ordinary SVM)
    -   L2 regularization + squared hinge loss (Command line option "-t 1")
    -   L2 regularization + logistic loss (Command line option "-t 7")
-   Ordinary LIBSVM/LIBLINEAR implementations do not accept instance weights. The ones that accept instance weights are distributed here: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances .

## How to run

### Prerequisites

The following software components are required:

-   make
-   gcc, g++
-   python (If you need Python wrapper)

If you try Python sample codes, the following Python libraries are required:

-   numpy
-   scipy
-   cvxpy
-   ecos
-   matplotlib
-   scikit-learn

For example of using Anaconda and you would like to create a new environment to run this, the following command will do this:

```
conda create -n liblinearWK -c conda-forge pip numpy scipy cvxpy matplotlib scikit-learn
conda activate liblinearWK
```

(Since this LIBLINEAR-Weights-Kernelized library is installed by pip, if you create an environment by conda, it is recommended to install pip by conda so that this library is installed only in this environment)

### 1. Build C++ library

Run `make` command in the top folder of this library files. Please confirm that it did not end with an error.

If you do not use Python library, the preparation is completed with this. Executable files `train` and `predict` will be produced, so please see command line options by running them.

### 2. Run the sample code to run it from command line

Move to `example_commandline` folder of this library files, and run the following command:

```
./linear_check.sh
```

Then the trained model parameters will be saved in the folder `example_commandline/linear_check_sh_output`.

### 3. Compare the resulted model parameters with CVXPY (Python required)

We can compare the result above with the ones computed by [CVXPY](https://www.cvxpy.org/) (optimizer for a large class of convex functions). In the folder `example_commandline`, run the following commands:

```
python linear_check.py
python linear_check_comparison.py
```

Then outputs like the followings will be shown. The "difference" means the L2-norm of trained model parameters between LIBLINEAR-Weights-Kernelized and CVXPY; Please confirm that the difference is sufficiently small.

```
linear_check_py_output/model-heart_scale-s7-t0-B1-W-1,1.1,2.log.primal
Size: (14,) (14,)
Difference: 3.98385354554017e-08
linear_check_py_output/model-heart_scale-s7-t0-B1.log.primal
Size: (14,) (14,)
Difference: 2.245077146169388e-08
```

### 4. Build Python library

Move to `python` folder, and run `make` command. Please confirm that it did not end with an error.

### 5. Install Python library

In the `python` folder, confirm that current Python environment is the one to install LIBLINEAR-Weights-Kernelized, and run `pip install -e .` command. (It is recommended to create a separate environment for pip and then install it.)

If the command did not end with an error, then confirm the installation by running the Python code `from liblinear import liblinear, liblinearutil, commonutil`. (It is easy to try in Python interactive mode.)

### 6. Run the sample code called from Python

Move to `example_python` folder, and run the following command:

```
python comparison_CVXPY_LIBLINEAR.py ../splice_scale 0.1 2 1
```

This will train SVM for the data file "../splice_scale" with the regularization strength 0.1, weights for positive instances 2 and for negative instances 1. Then we compare it with the implementation by [CVXPY](https://www.cvxpy.org/) the trained model parameters in the computation time and the model parameters.

Running this, it will plot the model parameters trained by this implementation and CVXPY, in the ascending order of the ones by this implementation. We will find that they are almost the same.

The computation times in the author's computer was as follows.

|Kernel|Optimizer for:|Training by CVXPY (s)|Training by This implementation (s)|
|:----:|:--------------:|------:|-----:|
|Linear|Primal variables| 1.1540|0.1979|
|Linear|Dual variables  | 1.5910|1.4058|
|RBF   |Dual variables  |12.4578|0.0641|

(Note that the optimizer for primal variables can be used only for linear kernel.)

## Appendix: The "intercept" in the formulation of SVM for this implementation

(See also: [Official document of LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf) and [Official document of LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf))

Suppose a linear prediction model for binary classifications, that is, given an outcome variable $y\in\\\{-1, +1\\\}$ and its explanatory variable $\boldsymbol{x}\in\mathbb{R}^k$, we would like to predict $y$ from $\boldsymbol{x}$ by a vector $\boldsymbol{\gamma}\in\mathbb{R}^k$ and a scalar called the **intercept** $b\in\mathbb{R}$ as

> $y\approx \mathrm{sign}(\boldsymbol{x}^\top\boldsymbol{\gamma} + b)$.

Given a $n$-instance training dataset $[y_1, y_2, \dots, y_n] \in\mathbb{R}^n$ and $X\in\mathbb{R}^{n\times k}$ and instance weights $[w_1, w_2, \dots, w_n] \in\mathbb{R}_{\geq 0}^n$, we conduct the training of $\boldsymbol{\gamma}$ and $b$ as follows:

> $\mathrm{argmin}\_{\boldsymbol{\gamma}\in\mathbb{R}^d, b\in\mathbb{R}} C\sum\_{i=1}^n w_i \ell(y\_i, X\_{i:}\boldsymbol{\gamma} + b) + \rho(\boldsymbol{\gamma})$,

where $\ell: \mathbb{R}\times\mathbb{R}\to\mathbb{R}\_{\geq 0}$ is a *loss function*, $\rho: \mathbb{R}^k\times\mathbb{R}\to\mathbb{R}\_{\geq 0}$ is a *regularization function*, and $C>0$ is a hyperparameter to control the strength of the regularization.

Here, we may use an alternative formulation. Let $d := k+1$, $\boldsymbol{\beta}\in\mathbb{R}^d$ and

$$\bar{X} :=
\begin{bmatrix}
& X &
\\\\\\hdashline
1 & \cdots & 1
\end{bmatrix}
\in\mathbb{R}^{n\times d}.$$

Then the training is conducted as

> $\mathrm{argmin}\_{\boldsymbol{\gamma}\in\mathbb{R}^d, b\in\mathbb{R}} C\sum_{i=1}^n w\_i \ell(y\_i, \bar{X}_{i:}\boldsymbol{\beta}) + \bar{\rho}(\boldsymbol{\beta})$.

Here, $\rho: \mathbb{R}^d\times\mathbb{R}\to\mathbb{R}_{\geq 0}$ is a regularization function for $d$-dimensional vector.

From the first term, we can interpret that

$$\boldsymbol{\beta} \approx
\begin{bmatrix} \boldsymbol{\gamma} \\\\ b
\end{bmatrix},$$

that is, the last element of $\boldsymbol{\beta}$ works as the intercept.  
The difference of the formulation from the previous one is that **the penalty by the regularization function is imposed on whole $\boldsymbol{\beta}$, that is, also on the intercept**. In usual data analysis we do not impose the penalty on the intercept, but in our setup we need to impose the penalty on the intercept to quantify the change of the model parameters when a part of dataset values are removed or altered.
