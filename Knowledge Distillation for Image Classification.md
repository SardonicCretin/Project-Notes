### Knowledge Distillation

Two models :  1. Teacher Model (Larger model) 2. Student Model (Distilled model)

Teacher model is used to supervise the training of the student model

Goal: Make the Student model match the class probability distribution / confidence of Teacher model (output probability matching). Can also match intermediate weights, intermediate feature maps, intermediate attention maps, etc.)

During Training:
1. Data sample forward passes to both the Teacher and Student models
2. Calculate the loss between the outputs of both models | Loss correlates to if prediction is correct AND what we want them to match 
3. Minimize the loss with backpropagation 
Dataset for distillation may be different from the dataset for training Teacher model, by which it is called *Transfer Set*

![Pasted image 20250211120423.png](Pasted%20image%2020250211120423.png)

For teacher and student network to confer with one another, need a loss function that does not just depend on correct prediction, but also effected by confidence values (output probability matching)

**Usual Hyperparameters:**
1. Temperature
2. Alpha
#### Temperature

The probabilities/confidence (softmax of output logits) output from the teacher network may have very small numbers (ex 0.999999999 for class A, 0.0000000001 for class B) over a small distribution (ex very few with super high or super low confidence)

To help, temperature component is added to "spread out" the bell curve and give the student model a wider, more varied range of values to work with

![666x522](Pasted%20image%2020250211113528.png)
![Pasted image 20250211121400.png](Pasted%20image%2020250211121400.png)

#### Loss Function
Two loss functions 
	1. One for Classification Accuracy -> Can be Cross Entropy
	2. One for Confidence Distribution Similarity -> KL Divergence

**Kullback-Leibler Divergence (KL Divergence)**
![Pasted image 20250211142000.png](Pasted%20image%2020250211142000.png)
![Pasted image 20250211141713.png](Pasted%20image%2020250211141713.png)

#### Alpha α

How much do we want the distillation loss to weigh more than the classifier loss?
At a range of 0 to 1, therefore α = 0.5 means equal weightage
### Cross Distillation

Cross-architecture knowledge distillation
	Transformer -> CNN
![Pasted image 20250211144527.png](Pasted%20image%2020250211144527.png)

### Multi-modal Distillation

Cross-modal knowledge distillation
	Lidar -> Camera | NDVI -> Camera
![Pasted image 20250211144152.png](Pasted%20image%2020250211144152.png)


___
Potential Ideas for Project: -
What needs guidance?
	attention map -> need to teach the model to look specifically at lesions

What if we do cross distillation of object detection -> image classification to improve the attention map? Few show object detection -> Image classifier to accelerate learning?

Classifier to Detector Distillation is more common (and more useful probably):
https://proceedings.neurips.cc/paper/2021/file/082a8bbf2c357c09f26675f9cf5bcba3-Paper.pdf

Low resolution teacher student learning
![Pasted image 20250211125831.png](Pasted%20image%2020250211125831.png)
https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720622.pdf