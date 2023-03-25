# SVM

## Project Description
Goal: Implement soft-SVM with the SMO algorithm for classification.

Functions to be implemented are

  – problem1.py: linear and Gaussian kernel functions, the hinge loss function,

  – problem2.py: dual and primal objective functions, decision function of SVM using the dual variables,

  – problem3.py: training of SVM and test prediction accuracy. You will use the functions defined in problem 1 and 2 to simplify your codes.

More detailed instructions are given in the comments of the functions.

• In files problem∗.py (∗ = 1, . . . , 3), there are functions for you to fill out with your codes.

• Files test∗.py (∗ = 1, . . . , 3) will unit-test the correctness of your implementations in the corresponding
problem∗.py files. For example, after you implement problem1.py file, run

nosetests -v test1

to test the correctness of your implementation of problem1.py. Note that passing the tests does not
mean your implementations are entirely correct: the test can catch only a limited number of mistakes.

If you use Anaconda (highly recommended), there should NOT be any packages you need to install.
Otherwise, you will need to install nosestest for such unit test.

• We provide simple visualization of SVM models by plotting training data, the decision boundary, and
the margins in the notebook:
src/svm_visualization.ipynb

You don’t need to plot the training/test loss during training, as they are output to the terminal when
we run problem4.py.

• There is no need to add/modify codes in svm visualization.ipynb or problem4.py
