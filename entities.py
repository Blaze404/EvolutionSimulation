# imports
import random
import math
import numpy as np
import helper
import neural_nets
from copy import deepcopy
import torch
import torch.optim as optim

class Entity:
    def __init__(self, entity_type: str, screen_size: int) -> None:
        self.entity_type: str = entity_type
        self.color = (128, 128, 128)

        self.default_mobility = 10
        self.default_visibility = 50
        self.default_size = 100

        self.minimum_mobility = 1
        self.minimum_visibility = 1
        self.minimum_size = 1

        self.maximum_mobility = 30
        self.maximum_visibility = screen_size
        self.maximum_size = 250

        self.mobility = 10
        self.size = 100
        self.visibility = 50

        self.total_energy = 100
        self.initial_total_energy = self.total_energy
        self.hungry_since = 0
        self.hunger_capacity = 3

        ## neural network stuff
        self.brain = None
        self.optimizer = None
        self.magnitude = None
        self.angle = None

        self.screen_size = screen_size
        self.entity_radius = 4

        self.age = 0
        self.reproduce_every = 2
        self.kids = 0

        # location stuff
        self.position = (random.randint(0, screen_size), random.randint(0, screen_size))
        self.color = (128, 128, 128)
        self.current_angle = float(random.randint(0, 359))
        self.generate_color()

        self.energy_usage_per_movement = 50
        self.mutation_rate = 0.001  # percent
        self.maximum_age = 10
        self.reproduction_energy = 95
        self.reproduction_success_probability = 0.75
        self.reproduction_spontaneous_probability = 0.25
        self.over_age_death_probability = 0.95
        self.minimum_energy_required_for_reproduction = self.reproduction_energy // 2


        self.was_in_opportunity = False
        self.was_able_to_eat = False

    def calculate_energy_usage(self, movement, distance) -> float:
        # return 0
        usage = self.energy_usage_per_movement

        # if self.mobility < self.default_mobility:
        #     multiplier = (self.mobility / self.default_mobility)
        # else:
        #     multiplier = (self.mobility / self.default_mobility) ** 1.5
        # if self.size < self.default_size:
        #     multiplier = multiplier * (self.size / self.default_size)
        # else:
        #     multiplier = multiplier * ((self.size/self.default_size)**2.5)
        # if self.visibility < self.default_visibility:
        #     multiplier = multiplier * (self.visibility / self.default_visibility)
        # else:
        #     multiplier = multiplier * ((self.visibility/self.default_visibility)**1.3)
        # multiplier = math.log(self.size) ** 3
        # multiplier = 1

        multiplier = movement / self.mobility

        if multiplier == 0:
            print("## ALERT: Multiplier is 0 for {}".format(self.entity_type))
        multiplier = max(0.25, multiplier)

        energy_used = usage*multiplier*distance

        energy_used = energy_used + self.size ** (3/10) * multiplier + self.size ** (1/10)

        energy_for_existing = self.size ** 0.65

        energy_used = energy_used + energy_for_existing
        energy_used = round(energy_used)
        # message = "Energy used: {}. Energy remaining: {}. Multiplier used: {}. Size: {}. Mobility: {}. Visibility: {}. Distance: {}".format(
        #     energy_used, self.total_energy - energy_used, multiplier, self.size,
        #     self.mobility, self.default_visibility, distance)
        # print(message)
        # message = "Default mobility: {}. Default size: {}. Default Visibility: {}. Movement: {}".format(
        #     self.default_mobility, self.default_size, self.default_visibility, movement
        # )
        # print(message)
        return energy_used

    def generate_color(self) -> None:
        r = (self.mobility / self.maximum_mobility) * 255
        r = min(max(round(r), 0), 255)

        g = (self.size / self.maximum_size) * 255
        g = min(max(round(g), 0), 255)

        b = (self.visibility / self.maximum_visibility) * 255
        b = min(max(round(b), 0), 255)

        self.color = (r, g, b)

    def set_opportunity(self, opp=True):
        self.was_in_opportunity = opp

    def set_params(self, params):
        mobility = params.get('mobility', self.default_mobility)
        size = params.get('size', self.default_size)
        visibility = params.get('visibility', self.default_visibility)

        mobility = min(max(self.minimum_mobility, mobility), self.maximum_mobility)
        size = min(max(self.minimum_size, size), self.maximum_size)
        visibility = min(max(self.minimum_visibility, visibility), self.maximum_visibility)

        self.mobility = mobility
        self.size = size
        self.visibility = visibility

        rp = (random.randint(0, self.screen_size), random.randint(0, self.screen_size))
        position = params.get('position', rp)

        self.generate_color()
        self.position = position

        brain = params.get('brain', None)
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

    def next_move(self, inputs: list) -> tuple:
        # print("For {}. Input size: {}".format(self.entity_type, len(inputs)))
        # print("Inputs: ", inputs[:-4])
        input_list = inputs
        input_tensor = torch.tensor(input_list, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0)
        self.brain.eval()
        angle, distance = self.brain(input_tensor)
        self.magnitude = distance
        self.angle = angle
        # angle, distance = output.detach().numpy().flatten().tolist()
        angle, distance = angle.item(), distance.item()
        # angle = random.random()
        # distance = random.random()
        # print("Output Angle: {}. Output Distance: {}".format(angle, distance))
        return angle, distance

    def check_if_mutate(self):
        mutate = helper.calculate_yes_no_probability(self.mutation_rate)
        return mutate

    def get_mutation_values(self):
        mobility_mutation = 0
        if self.check_if_mutate():
            mu = self.mobility  # mean and standard deviation
            sigma = mu * 0.25
            s = np.random.normal(mu, sigma, 1)
            mobility_mutation = round(s[0])

        size_mutation = 0
        if self.check_if_mutate():
            mu = self.size  # mean and standard deviation
            sigma = mu * 0.25
            s = np.random.normal(mu, sigma, 1)
            size_mutation = round(s[0])

        visibility_mutation = 0
        if self.check_if_mutate():
            mu = self.visibility  # mean and standard deviation
            sigma = mu * 0.25
            s = np.random.normal(mu, sigma, 1)
            visibility_mutation = round(s[0])

        return mobility_mutation, size_mutation, visibility_mutation

    def check_if_about_to_die(self):
        # return False
        if self.total_energy + self.size <= 0:
            return True
        hc = self.hunger_capacity
        if self.hungry_since > hc:
            possibility = helper.calculate_yes_no_probability(0.9)
            if possibility:
                return True
        if self.age > self.maximum_age:
            if self.entity_type == 'predator':
                if self.total_energy > 1.2 * self.initial_total_energy:  # starting energy
                    # why? because this is a very successful predator which can live longer
                    return helper.calculate_yes_no_probability(self.over_age_death_probability/1.25)
            return helper.calculate_yes_no_probability(self.over_age_death_probability)
        return False

    def reproduce(self):
        raise NotImplementedError("Please Implement this reproduce method")

    def check_eligibility_for_reproduction(self):
        # return False
        if self.total_energy + self.size <= self.minimum_energy_required_for_reproduction:
            return False
        if (helper.calculate_yes_no_probability(self.reproduction_spontaneous_probability)
                and self.age > self.reproduce_every):
            return True
        if self.age % self.reproduce_every == 0:
            return helper.calculate_yes_no_probability(self.reproduction_success_probability)

    def move(self, angle: float, distance: float):
        self.age += 1
        self.hungry_since += 1
        self.was_able_to_eat = False
        angle = 360 * angle
        angle_radians = math.radians(angle)
        center = self.position

        distance_moved = self.mobility * distance

        energy_usage = self.calculate_energy_usage(distance_moved, distance)
        self.total_energy -= energy_usage

        movement_penalty = self.size ** (3/10)
        movement_penalty = movement_penalty / 100
        movement_penalty = 1 - movement_penalty
        distance_moved = distance_moved * movement_penalty

        # print("Distance Moved: {}. Distance: {}. Movement Penalty: {}".format(distance_moved, distance, movement_penalty))

        dx = distance_moved * math.cos(angle_radians)
        dy = distance_moved * math.sin(angle_radians)
        # print("Updating position by: x - {} and y - {}. Got angle: {}".format(dx, dy, angle))
        end_pos_x = round(center[0] + dx)
        end_pos_y = round(center[1] - dy)

        if end_pos_x < 0:
            end_pos_x = 0
        if end_pos_y < 0:
            end_pos_y = 0
        if end_pos_x > self.screen_size:
            end_pos_x = self.screen_size
        if end_pos_y > self.screen_size:
            end_pos_y = self.screen_size

        self.position = (end_pos_x, end_pos_y)
        self.current_angle = angle



        # self.generate_color()

class Prey(Entity):
    def __init__(self, screen_size: int) -> None:
        self.angle_of_vision = 360.0  # in degrees

        super().__init__("prey", screen_size)
        self.generate_brain()
        self.gazing_food = int(self.initial_total_energy * 0.55)
        self.food_scarcity_fight_probability = 0.01
        self.total_energy = self.total_energy * 1.1
        self.attack_escape_probability = 0.1

    def generate_brain(self) -> None:
        brain = neural_nets.PreyNN()
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01)

    def reproduce(self):
        self.kids += 1
        new_ent = Prey(self.screen_size)
        params = dict()

        mobility_mutation, size_mutation, visibility_mutation = self.get_mutation_values()

        params['mobility'] = self.mobility + mobility_mutation
        params['size'] = self.size + size_mutation
        params['visibility'] = self.visibility + visibility_mutation


        params['color'] = self.color
        pos = self.position
        if pos[0] < self.screen_size / 2:
            new_x = pos[0] + self.entity_radius
        else:
            new_x = pos[0] - self.entity_radius
        if pos[1] < self.screen_size / 2:
            new_y = pos[1] + self.entity_radius
        else:
            new_y = pos[1] - self.entity_radius

        params['position'] = (new_x, new_y)

        new_brain = deepcopy(self.brain)
        neural_nets.mutate_weights(new_brain, mutation_rate=self.mutation_rate)
        params['brain'] = new_brain

        new_ent.set_params(params)
        self.total_energy -= self.reproduction_energy
        return new_ent

    def gaze(self, total_preys: int, supported_preys: int):
        if total_preys <= supported_preys:
            total_food = self.gazing_food * supported_preys
            food_per_prey = round(total_food / total_preys, 2) * 0.95
            self.total_energy += max(self.gazing_food, food_per_prey)
            self.hungry_since = 0
            return
        else:
            # they need to ration
            # if this prey fought for food:
            if supported_preys == 0:
                supported_preys = 1
            fsfp = self.food_scarcity_fight_probability + (((total_preys-supported_preys)**0.65) / supported_preys)
            fsfp = round(min(0.9, fsfp), 3)
            if helper.calculate_yes_no_probability(fsfp):
                if helper.calculate_yes_no_probability(0.001):
                    print("{}: Fight probability".format(fsfp))
                self.total_energy -= 0.99 * abs(self.initial_total_energy)
                return
            total_food = self.gazing_food * supported_preys
            food_per_prey = round(total_food / total_preys, 2)
            self.total_energy += food_per_prey
            self.hungry_since = 0
            self.was_able_to_eat = True
            # print("Food per day: {} instead of {}".format(food_per_prey, self.gazing_food))
            return

    def feedback(self, negative=False):
        opp = self.was_in_opportunity
        self.was_in_opportunity = False

        if negative:
            reward = -1
        else:
            reward = 1 if opp else 0.3
        loss_angle = -torch.log(self.angle / 360) * reward  # Normalize angle for log probability
        loss_magnitude = -torch.log(self.magnitude) * reward

        total_loss = loss_angle + loss_magnitude
        # print(total_loss)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def get_kill_success(self, predator):
        if self.size > predator.size * 1.75:
            return helper.calculate_yes_no_probability(0.6)
        if self.total_energy + self.size > (predator.total_energy + predator.size) * 1.75:
            return helper.calculate_yes_no_probability(0.5)
        return helper.calculate_yes_no_probability(1-self.attack_escape_probability)


class Predator(Entity):
    def __init__(self, screen_size: int) -> None:
        super().__init__("predator", screen_size)
        self.angle_of_vision = 100.0  # in degrees
        self.visibility = 100
        self.default_visibility = 100
        self.generate_color()
        self.predator_hitting = 10
        self.max_preys_hit = 4
        # self.reproduce_every = self.reproduce_every + 1
        self.energy_usage_per_movement = self.energy_usage_per_movement + 5
        self.reproduction_energy = self.reproduction_energy
        self.generate_brain()

    def generate_brain(self) -> None:
        brain = neural_nets.PredatorNN()
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

    def eat(self) -> None:
        # sometimes a predator can eat multiple preys, so if it has eaten too much don't increase energy
        if self.total_energy > 2*self.initial_total_energy:
            return
        self.total_energy += self.initial_total_energy * 0.8
        self.hungry_since = 0

    def predator_pressure(self, total_predators: int, total_preys: int):
        if total_preys >= total_predators:
            # no pressure
            return

        # chances of fighting will be developed
        fight_chance = ((total_predators - total_preys) ** 0.65) / total_predators
        fight_chance = round(min(0.9, fight_chance), 2)
        if helper.calculate_yes_no_probability(fight_chance):
            if helper.calculate_yes_no_probability(0.001):
                print("{}: Fight probability - Predator".format(fight_chance))
            self.total_energy -= 0.99 * self.initial_total_energy

    def reproduce(self):
        self.kids += 1
        new_ent = Predator(self.screen_size)
        params = dict()

        mobility_mutation, size_mutation, visibility_mutation = self.get_mutation_values()

        params['mobility'] = self.mobility + mobility_mutation
        params['size'] = self.size + size_mutation
        params['visibility'] = self.visibility + visibility_mutation

        params['color'] = self.color
        pos = self.position
        if pos[0] < self.screen_size / 2:
            new_x = pos[0] + self.entity_radius
        else:
            new_x = pos[0] - self.entity_radius
        if pos[1] < self.screen_size / 2:
            new_y = pos[1] + self.entity_radius
        else:
            new_y = pos[1] - self.entity_radius

        params['position'] = (new_x, new_y)

        new_brain = deepcopy(self.brain)
        neural_nets.mutate_weights(new_brain, mutation_rate=self.mutation_rate)
        params['brain'] = new_brain

        new_ent.set_params(params)
        reproduction_energy = self.reproduction_energy + (self.size ** 0.25)
        self.total_energy -= reproduction_energy
        return new_ent

    def feedback(self, negative=False):
        opp = self.was_in_opportunity
        self.was_in_opportunity = False

        if helper.calculate_yes_no_probability(0.5):
            return

        if not negative:
            reward = 1 if opp else 0.3
        else:
            reward = -1
        loss_angle = -torch.log(self.angle / 360) * reward  # Normalize angle for log probability
        loss_magnitude = -torch.log(self.magnitude) * reward

        total_loss = loss_angle + loss_magnitude
        # print(total_loss)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def got_hit(self, count):
        self.total_energy -= self.predator_hitting * min(count, self.max_preys_hit)
