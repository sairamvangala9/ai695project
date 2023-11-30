# Project

Group Name:DMSR\
Student Names: Divya Malyala (G01390473), Sai Ram Vangala (G01373045), Vinitha Puligorla (G01397891)

# Example

Lets consider the below DNN example

![img_8.png](img_8.png)


Lets assume that the ouput for this DNN is f<20


# Naive Interval Propagation

Input layers nodes = x,y
Hidden layers nodes = A,B
Output layers node = f

Step1: Hidden layer 1 evaluation
```commandline
lb = lower bound
ub = upper bound
```

![img_9.png](img_9.png)

Here we get output interval as [0,22], which is a Safety violation

# Symbolic Interval Analysis

In this approach, instead of the intervals, input nodes are considered as variables and formula for the output nodes are evaluated by considering the weights <br>
Input interval values are applied to the formula generated for the output nodes

![img_10.png](img_10.png)

In this case we get output interval as [6,16]. It was better than naive approach but does over approximation

# Iterative Interval Refinement using Bisection

Iterative Interval Refinement has same process as of Naive interval propagation approach in addition with bisection<br>
An input interval is bisected into 2 parts and then the outputs are validated. The final output interval as calculated by performing Union operation<br>
We choose the interval to bisect based on smear function, which is dependent on input range width and absolute gradient In this case, it would be node 'y' interval, so bisecting it, gives [1,3] and [3,5].<br>
Intervals for Nodes, A,B,f considering each bisected intervals are calculated individually<br>

1st Iteration for the hidden layer nodes
![img_11.png](img_11.png)

2nd Iteration for the output layer nodes
![img_12.png](img_12.png)

Here we still did not get safety as the output interval resulted in [-4,21] <br>

# More bisection of intervals

1st iteration for hidden layer nodes
![img_13.png](img_13.png)
![img_15.png](img_15.png)
2nd iteration for output layer nodes
![img_14.png](img_14.png)

If we perform union for the f intervals we get
```
[5,18] U [3,17] U [1,15] U [-1,13] = [-1,18]
```
Which is safe. Hence we get safe intervals with ReluVal by implementing bisection
