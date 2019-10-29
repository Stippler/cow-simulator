# Cow Simulator

## References
For further reference look at the *references.bib* file.

[Multi-agent actor-critic for mixed cooperative-competitive environments](https://arxiv.org/pdf/1706.02275.pdf)

[Control a cart](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

[Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/content/114/13/3521.abstract)

[Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/pdf/1909.07528.pdf)

[Youtube Playlist I got my inspiration from](https://www.youtube.com/watch?v=xukp4MMTTFI&list=PL58qjcU5nk8u4Ajat6ppWVBmS_BCN_T7-&index=1 "Youtube Playlist Inspiration")

## Topic
The Topic is Reinforcement learning.

## Type
The type of my project is a mixture of **Bring your own data** and **Bring your own method**. I want to create my own environment that has not yet been used in another project. Therefore I assume that I have to bring my own method as well, as I have not found an existing one that has the same sensors and actuators.

## Summary
### Description

As an applied deep learning project I want to build a simulation consisting of cows, gras and soil. The white background represents the soil, the brown circle represents the cow, the green circle represents the gras and the red lines represent the rays, each cow can send out to percept its environment.

![figure1](environment.png)

A cow consists of 5 elements:

* A body, basically a brown circle.
* An energy level, if its energy level reaches 0 the cow dies, if it reaches an energy level that is high enough it reproduces.
* A perceptor, that sends out rays. In every update cycle that perceptor sends out it rays. If a ray collides with an object, it returns the color, if not white gets returned.
* Actuators to control steering and speed.
* An artificial Neural Network that gets the colors of the perceptor as input and uses the actuators to make actions.

The purpose of the gras is to replenish the energy of the cow. When a cow collides with gras, the gras energy level decreases while the cows energy level goes up. The Neural Network is trained by Reinforcement learning, it gets a reward if the cow replenishes energy.

The goal of the simulation is to implement a flexible environment for trying out different Reinforcement learning approaches. 

### Dataset
There is no real dataset. I have to implement the environmet and make it possible to searlize the actions the agents make in order to analyze their behaviour.

### Work-Breakdown structure
