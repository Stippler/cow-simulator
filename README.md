# Cow Simulator

## References

### scientific papers

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/content/114/13/3521.abstract)

### articles/tutorials that were used in preparation

[Control a cart](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

[Youtube Playlist I got my inspiration from](https://www.youtube.com/watch?v=xukp4MMTTFI&list=PL58qjcU5nk8u4Ajat6ppWVBmS_BCN_T7-&index=1 "Youtube Playlist Inspiration")

[Multi-agent actor-critic for mixed cooperative-competitive environments](https://arxiv.org/abs/1706.02275)

[Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528)


## Topic
This project includes different Reinforcement Learning strategies in a dynamic, multi-agent environment. <!-- TODO: define it more precisly-->

## Type
The type of this project is **Bring your own data** for reinforcement learning projects, because it provides a new environment for reinforcement learning strategies. Additionally it includes basic neural networks for every actor and learning algorithms for them. It is also partly **Beat the classics** as it is tested against rule based systems with path finding algorithms. 

## Summary
### Description

This project consists of a simulation that simulates a partially observable, multi-agent, stochastic, dynamic, continuous in space, discrete in time and partly unknown (missing knowledge about laws of physics) environment.

There are two different groups of actors that can interact consciously with the environment: cows and wolves. 
Additionally, there is another entity called grass. Each actor has a certain energy level. If the energy level of a cow or a wolf drops below zero, it dies. 
Cows and wolves have perceptors in order to obtain a partial observation of the environment.

The following figure shows a visualisation of the actors, the green circle represents grass, the brown circle represents a cow and the blue circle represents a wolf. The little circles represent their perceptors.

![figure1](actors.png)

If an actor wants to perceive its environment, its perceptor sends out rays with a limited reach. The rays return the color of the actor/perceptor they first intersect with or black if they do not intersect with another actor. The next figure shows a visualisation of the rays. The color of the rays show the value they return.

![figure1](perceptor.png)

Every entity has energy, if the energy drops below zero the entity vanishes. The cow gets energy by touching grass, the wolf by touching cows. Each entity loses energy by touching its counterpart or moving around. The goal of each actor is to obtain as much energy as possible. Grass gets more energy if it is left alone.

To implement the actors' AI the approach of (Proximal Policy Optimization Algorithms)[https://arxiv.org/abs/1707.06347] is primarily used. Furthermore, [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/content/114/13/3521.abstract) is the basis for teaching the actors to achieve sequential tasks, such as running away from predators after eating.

### Dataset
There is no real dataset. The project implements the environment, gives an visualisation of the state of the world and implements rule based actors.

### Work-Breakdown structure

| Individual Task &nbsp;                                     | Time estimate &nbsp; | Time used |
|------------------------------------------------------------|----------------------|-----------|
| research topic and first draft                              | 5h                   | 8h        |
| building environment                                       | 10h                  |           |
| designing and building an appropriate network &nbsp;&nbsp; | 20h                  |           |
| fine-tuning that network                                    | 10h                  |           |
| building an application to present the results             | 5h                   |           |
| writing the final report                                    | 10h                  |           |
| preparing the presentation of the project                  | 5h                   |           |

