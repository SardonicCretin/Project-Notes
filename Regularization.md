=Controlling the capacity of the neural network to prevent overfitting, allowing the model to generalize

In an ideal world, the max activations of the neuron would result in a generic pattern that defines the class (e.x, wheels = car, right angle = L). In reality, it results in abstract, uninterpretable patterns.

Note: Gradient Ascent = find values that maximally activates a neuron

![[Pasted image 20250210125614.png]]

We can improve this by adding regularization techniques, to reduce the "memorization" during training.

**Dropout**
During training, each neuron is only active with some assigned probability

![[Pasted image 20250210124137.png]]
