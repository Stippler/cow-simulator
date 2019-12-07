from deepcow.constant import *
from deepcow.actions import *

from pygame.math import Vector2
import pygame


class Entity(object):
    def __init__(self,
                 position,
                 radius=DEFAULT_RADIUS,
                 initial_energy=DEFAULT_INITIAL_ENERGY,
                 color=(0, 0, 0)):
        self.position = position
        self.radius = radius
        self.energy = initial_energy
        self.color = color

    def draw(self, screen):
        x = int(self.position.x)
        y = int(self.position.y)
        pygame.draw.circle(screen, self.color, (x, y), self.radius)


class Agent(Entity):
    def __init__(self,
                 position,
                 velocity,
                 max_speed=DEFAULT_MAX_SPEED,
                 acceleration=DEFAULT_ACCELERATION,
                 radius=DEFAULT_RADIUS,
                 initial_energy=DEFAULT_INITIAL_ENERGY,
                 mass=DEFAULT_MASS,
                 elasticity=DEFAULT_ELASTICITY,
                 color=(0, 0, 0)):
        super(Agent, self).__init__(position, radius, initial_energy, color)
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.velocity = velocity
        self.mass = mass
        self.elasticity = elasticity

    def update_position(self, delta):
        speed = self.velocity.magnitude()
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        self.position += (self.velocity * delta)

    def calculate_agents_collisions(self, agents):
        # collision with other agents
        for agent in agents:
            if self != agent:
                # vector pointing from self to the agent
                self_to_agent_vec = agent.position - self.position

                # distance between self and agent
                distance = self_to_agent_vec.magnitude()
                total_radius = self.radius + agent.radius

                # stop check if agents are not intersecting
                if distance >= total_radius:
                    continue

                # normalize self_to_agent_vec or set it to x=1 if the agents have the exact same position
                # to avoid division by zero
                if distance == 0:
                    self_to_agent_vec.x = 1
                else:
                    self_to_agent_vec /= distance

                # calculate the overlap of the agents
                overlap = total_radius - distance
                total_mass = self.mass + agent.mass

                # reset their position based on their mass so they do not overlap anymore
                self.position += -((self.mass / total_mass) * overlap * self_to_agent_vec)
                agent.position += ((agent.mass / total_mass) * overlap * self_to_agent_vec)

                # calculate the new velocities
                self_to_agent_vec *= self_to_agent_vec.dot(agent.velocity - self.velocity) / total_mass

                # change speed based on elasticity and mass
                self.velocity = self.velocity + self_to_agent_vec * (2 * agent.mass * (1 + self.elasticity))
                agent.velocity = agent.velocity + self_to_agent_vec * (-2 * self.mass * (1 + agent.elasticity))

    def calculate_border_collisions(self):
        # collision with border
        pos = self.position
        vel = self.velocity
        r = self.radius
        right_distance = GAME_WIDTH - (pos.x + r)
        left_distance = pos.x - r
        up_distance = pos.y - r
        down_distance = GAME_HEIGHT - (pos.y + r)
        if right_distance < 0:
            pos.x += right_distance
            vel.x *= -1
        elif left_distance < 0:
            pos.x -= left_distance
            vel.x *= -1
        if up_distance < 0:
            pos.y -= up_distance
            vel.y *= -1
        elif down_distance < 0:
            pos.y += down_distance
            vel.y *= -1

    def update_acceleration(self, delta, acceleration_action, rotation_action):
        if acceleration_action != AccelerationAction.NOTHING:
            acceleration_delta = self.acceleration * delta
            if acceleration_action == AccelerationAction.FORWARD:
                self.velocity += Vector2(0.0, -1.0) * acceleration_delta
            elif acceleration_action == AccelerationAction.BACKWARD:
                self.velocity += Vector2(0.0, 1.0) * acceleration_delta
            elif acceleration_action == AccelerationAction.LEFT:
                self.velocity += Vector2(-1.0, 0.0) * acceleration_delta
            elif acceleration_action == AccelerationAction.RIGHT:
                self.velocity += Vector2(1.0, 0.0) * acceleration_delta

        if rotation_action != RotationAction.NOTHING:
            pass
