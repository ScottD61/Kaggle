README.MD

Data: Used the random forest tutorial (attached .pdf) by bruschkov for separating timestamp to year, hours, and day of week variables. 

Least Squares method for fitting regression line to data and predict variable register in test set. 

Two models: Lease squares with nominal variables left as factor and factors converted to binary. 

To fix: Getting rank difficient fit for two different types of models after using predict(). 

Warning message:
In predict.lm(model1, Bint) :
  prediction from a rank-deficient fit may be misleading

To do: research ways to predict multiple nominal variables with >2 factors without multicollinearity warning. Results will not work because negative values of register exist. 

