### Part 1: **Convolutional** Neural Network

#### What is a convolution?

![Pasted image 20250210122036.png](Pasted%20image%2020250210122036.png)

A sliding "filter" that moves throughout the image, where we are looking if something exists within an image.  In other words, we ask:- What does each filter want to see in the image? What has each filter learned to look for?

The result of applying the filter on the input image / previous layer is called a **feature map**

### Part 2: Convolutional **Neural Network**

![Pasted image 20250211124331.png](Pasted%20image%2020250211124331.png)
#### What is a single layer?

![Pasted image 20250210121911.png](Pasted%20image%2020250210121911.png)

A 64 x 3 x 11 x 11 layer
	means: there are 64 "3 channel 11 x 11 filters", where each "filter" represents learnable weights for what to look for in the image when performing a convolution

#### What happens in multiple layers?

In reality, each layer feeds into the next layer, thus hard to visualize what the weights actually represent since the input of a layer depends on the previous layer AND it keeps going

![Pasted image 20250210120655.png](Pasted%20image%2020250210120655.png)

**Last Layer**
A fully connected layer, all neurons of the layer has a 1 to 1 connection with the neurons in the previous layer.
-> Interpret the feature maps into a final output

How do we visualize the last layer? 
t-SNE (t-distributed stochastic neighbor embedding) dimensionality reduction
-> Reduce the dimensionality of the vectors to 2D, gives us a graph of how the classes are separated
![Pasted image 20250210131950.png](Pasted%20image%2020250210131950.png)


**Output Layer**
Classifier; outputs probability distribution for each class
