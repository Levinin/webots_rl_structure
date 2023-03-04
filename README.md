# Webots with DDPG

A simple <a href="https://cyberbotics.com/">Webots</a> phototaxis environment set up for Reinforcement Learning.

The robot controller will house the model 'brain' while the 
supervisor will control the environment and reset everything after
each episode.

The RL algorithm used is <a href="https://arxiv.org/abs/1509.02971">DPPG</a>, implemented in pytorch.

## Current status

### Controllers

The supervisor and epuck controller are set up in the webots environment. Their configuration allows for 
episodic learning. The supervisor controls the episodes and resets the environment after each episode.
This allows for randomisation as required.

### Phototaxis Environment

The current environment is very simple with a small open area and a light in one corner. The light will 
activate the epuck's light sensors across most of the area. With a step reward of light activation, this environment 
will provide a nice reward gradient across most of the surface. 

The robot starts in a region which does not activate the light sensors and 
facing away from the light, so a little exploration is required to find the 
edge of the light.

### DDPG

In progress.

### General

Phototaxis is obviously easier with a BBR approach, however this provides a clean and simple environment to test 
RL implementations to make sure they do a little bit of searching and can learn.
