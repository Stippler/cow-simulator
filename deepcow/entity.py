from deepcow.constant import *
from deepcow.actions import *

from pygame.math import Vector2
from math import sqrt
import pygame
import numpy as np

WHITE = (255, 255, 255)


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
        radius = int(self.radius)
        pygame.draw.circle(screen, self.color, (x, y), radius)


class Agent(Entity):
    def __init__(self,
                 position,
                 velocity,
                 field_of_view=DEFAULT_FIELD_OF_VIEW,
                 ray_count=DEFAULT_RAY_COUNT,
                 ray_length=DEFAULT_RAY_LENGTH,
                 direction=Vector2(1.0, 0.0),
                 rotation_speed=DEFAULT_ROTATION_SPEED,
                 max_speed=DEFAULT_MAX_SPEED,
                 acceleration=DEFAULT_ACCELERATION,
                 radius=DEFAULT_RADIUS,
                 initial_energy=DEFAULT_INITIAL_ENERGY,
                 mass=DEFAULT_MASS,
                 elasticity=DEFAULT_ELASTICITY,
                 color=(0, 0, 0)):
        super(Agent, self).__init__(position, radius, initial_energy, color)
        self.field_of_view = field_of_view
        self.ray_count = ray_count
        self.ray_length = ray_length
        self.rotation_speed = rotation_speed
        self.direction = direction
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.velocity = velocity
        self.mass = mass
        self.elasticity = elasticity

    def percept(self, entities, screen=None):
        stop_angle = self.field_of_view / 2
        start_angle = -stop_angle
        delta_angle = self.field_of_view / self.ray_count
        angles = np.arange(start_angle, stop_angle + delta_angle, delta_angle)
        perceptions = []
        head_position = self.get_head_position()
        for angle in angles:
            # vector from ray start to ray end
            ray_direction_vec = self.direction.rotate(angle) * self.ray_length
            collisions = []
            for entity in entities:
                if entity == self:
                    continue
                # vector from entity position to ray start (i.d. head of self)
                entity_head_vec = head_position - entity.position

                # solve quadratic equation
                a = ray_direction_vec.dot(ray_direction_vec)
                b = 2 * entity_head_vec.dot(ray_direction_vec)
                c = entity_head_vec.dot(entity_head_vec) - self.radius ** 2
                discriminant = b * b - 4 * a * c
                if discriminant >= 0:
                    # ray hit the entity
                    discriminant = sqrt(discriminant)
                    t1 = (-b - discriminant) / (2 * a)
                    t2 = (-b + discriminant) / (2 * a)
                    if 0 <= t1 <= 1:
                        # impale or poke (e.g. cow looks at wolfe or vice versa)
                        collisions.append((ray_direction_vec * t1, entity.color))
                    elif 0 <= t2 <= 1:
                        # exit wound (e.g. cow eats grass, head center is inside of grass)
                        collisions.append((ray_direction_vec * t2, entity.color))

            collision_count = len(collisions)
            if collision_count >= 1:
                perception = collisions[0]
                if collision_count > 1:
                    # get closest element
                    distance_squared = perception[0].magnitude_squared()
                    for collision in collisions[1:]:
                        new_distance_squared = collision[0].magnitude_squared()
                        if new_distance_squared < distance_squared:
                            perception = collision
                            distance_squared = new_distance_squared
            else:
                # calculate vector that points to end of the ray
                end_point = head_position + ray_direction_vec
                # check if it is inside the game boundaries
                if 0 <= end_point.x <= GAME_WIDTH and 0 <= end_point.y <= GAME_HEIGHT:
                    perception = (ray_direction_vec, (255, 255, 255))
                else:
                    perception = (ray_direction_vec, (0, 0, 0))

            perceptions.append(perception)
            if screen is not None:
                start_x = int(head_position.x)
                start_y = int(head_position.y)
                end_position = head_position + perception[0]
                end_x = int(end_position.x)
                end_y = int(end_position.y)
                pygame.draw.line(screen, perception[1], (start_x, start_y), (end_x, end_y))
        return perceptions

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

                # stop calculation if agents are not intersecting
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
                agent.velocity = agent.velocity + self_to_agent_vec * (
                        -2 * self.mass * (1 + agent.elasticity))

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

    def perform_action(self, delta, action):
        if action != Action.NOTHING:
            acceleration_delta = self.acceleration * delta
            if action == Action.MOVE_FORWARD:
                self.velocity += Vector2(0.0, -1.0) * acceleration_delta
            elif action == Action.MOVE_BACKWARD:
                self.velocity += Vector2(0.0, 1.0) * acceleration_delta
            elif action == Action.MOVE_LEFT:
                self.velocity += Vector2(-1.0, 0.0) * acceleration_delta
            elif action == Action.MOVE_RIGHT:
                self.velocity += Vector2(1.0, 0.0) * acceleration_delta
            elif action == Action.ROTATE_CLOCKWISE:
                self.direction = self.direction.rotate(self.rotation_speed * delta)
            elif action == Action.ROTATE_COUNTER_CLOCK:
                self.direction = self.direction.rotate(-self.rotation_speed * delta)

    def get_head_position(self):
        return self.position + self.direction * self.radius

    def draw(self, screen):
        super().draw(screen)
        head_position = self.get_head_position()
        x = int(head_position.x)
        y = int(head_position.y)
        radius = int(self.radius / 2)
        pygame.draw.circle(screen, (0, 0, 0), (x, y), radius)