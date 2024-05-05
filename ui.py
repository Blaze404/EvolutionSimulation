import pygame
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pygame.locals import *
from collections import deque
import time

class EvolutionUI:
    def __init__(self, screen_size: int, total_ticks: int):

        self.is_running = True

        self.screen_size = screen_size
        self.screen = None
        self.entity_radius = 4
        # colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.default_size = 1000

        self.information_area = 700
        self.font = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

        self.display_legend = True
        self.max_graph_length = 125
        self.count_history_preys = deque(maxlen=self.max_graph_length)
        self.count_history_predators = deque(maxlen=self.max_graph_length)
        self.sum_history_entities = deque(maxlen=self.max_graph_length)

        self.color_red_average_prey = deque(maxlen=self.max_graph_length)
        self.color_green_average_prey = deque(maxlen=self.max_graph_length)
        self.color_blue_average_prey = deque(maxlen=self.max_graph_length)

        self.color_red_average_predator = deque(maxlen=self.max_graph_length)
        self.color_green_average_predator = deque(maxlen=self.max_graph_length)
        self.color_blue_average_predator = deque(maxlen=self.max_graph_length)

        self.history_append_probability = min(round((total_ticks ** 0.75) / total_ticks, 3) * 2, 1)
        self.epoch = 0
        self.start_time = time.time()

        self.text_start = 20
        self.text_row_height = 23

    def create_screen(self) -> pygame.Surface:
        pygame.init()
        screen = pygame.display.set_mode((self.default_size + self.information_area, self.default_size))
        pygame.display.set_caption('Evolution Simulation by blaze')
        self.screen = screen
        self.refresh_screen()

        font = pygame.font.Font('freesansbold.ttf', 20)
        self.font = font

        # charts
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9))
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3

        return screen

    def refresh_screen(self) -> None:
        self.screen.fill(self.white)
        pygame.display.flip()

    def generate_partition_line(self):
        line_start = (self.screen_size + self.entity_radius, 0)
        line_end = (self.screen_size + self.entity_radius, self.screen_size)
        pygame.draw.line(self.screen, self.black, line_start, line_end)

        horizontal_line_start = (0, self.screen_size + self.entity_radius)
        horizontal_line_end = (self.screen_size + self.entity_radius, self.screen_size)
        pygame.draw.line(self.screen, self.black, horizontal_line_start, horizontal_line_end)

        top_line_start = (0, 0)
        top_line_end = (self.screen_size, 0)
        pygame.draw.line(self.screen, self.black, top_line_start, top_line_end)

        left_line_start = (0, 0)
        left_line_end = (0, self.screen_size)
        pygame.draw.line(self.screen, self.black, left_line_start, left_line_end)



    def add_entities(self, preys, predators) -> None:

        # pygame stuff
        for event in pygame.event.get():
            if event.type == QUIT:
                # Clean up Pygame
                self.is_running = False

        # informatives

        text_epochs = self.font.render('Day: {}.'.format(self.epoch), True, self.black)
        self.epoch += 1
        text_epochs_rect = text_epochs.get_rect()
        text_epochs_rect_x = self.default_size + (self.information_area // 2)
        text_epochs_rect_y = self.text_start
        text_epochs_rect.center = (text_epochs_rect_x, text_epochs_rect_y)

        n_preys = len(preys)
        n_predators = len(predators)

        nm_prey = max(1, n_preys)
        nm_predator = max(1, n_predators)



        # add_to_chart = False
        # if np.random.choice([True, False], p=[self.history_append_probability, 1-self.history_append_probability]):
        add_to_chart = True
        self.count_history_preys.append(n_preys)
        self.count_history_predators.append(n_predators)
        self.sum_history_entities.append(n_preys + n_predators)

        self.screen.fill(self.white)


        # charts
        self.ax1.clear()
        self.ax1.plot(range(len(self.count_history_preys)), self.count_history_preys,
                      color='green', label='Preys')  # Example data
        self.ax1.plot(range(len(self.count_history_predators)), self.count_history_predators,
                      color='red', label='Predators')
        self.ax1.set_title('Count of prey/predators')
        # self.ax1.set_xlabel('Data point')
        self.ax1.set_ylabel('No of entities')
        # self.ax2.plot(range(len(self.sum_history_entities)), self.sum_history_entities, color='blue')  # Example data
        # self.ax2.set_title('Count of total prey/predators')
        # self.ax2.set_xlabel('Data point')
        # self.ax2.set_ylabel('No fof entities')
        canvas = FigureCanvas(self.fig)
        canvas.draw()

        self.screen.blit(text_epochs, text_epochs_rect)

        color_red_sum_prey = 0
        color_green_sum_prey = 0
        color_blue_sum_prey = 0
        total_age_prey = 0
        max_age = 0
        prey_kids = 0
        mature_preys_count = 0
        for prey in preys:
            center = prey.position
            color = prey.color
            er = prey.entity_radius
            total_age_prey += prey.age
            max_age = max(prey.age, max_age)
            if prey.age >= prey.reproduce_every:
                prey_kids += prey.kids
                mature_preys_count += 1
            # charts
            color_red_sum_prey += color[0]
            color_green_sum_prey += color[1]
            color_blue_sum_prey += color[2]

            internal_angle = prey.current_angle
            internal_angle_radians = math.radians(internal_angle)
            pygame.draw.circle(self.screen, color, center, er)

            dx = 1.5 * er * math.cos(internal_angle_radians)
            dy = 1.5 * er * math.sin(internal_angle_radians)

            end_pos_x = round(center[0] + dx)
            end_pos_y = round(center[1] - dy)

            end_pos = (end_pos_x, end_pos_y)

            pygame.draw.line(self.screen, self.blue, center, end_pos)

        color_red_sum_predator = 0
        color_green_sum_predator = 0
        color_blue_sum_predator = 0
        predator_kids = 0
        total_age_predator = 0
        mature_predators_count = 0
        for predator in predators:
            center = predator.position
            color = predator.color
            er = predator.entity_radius
            total_age_predator += predator.age
            max_age = max(predator.age, max_age)
            if predator.age >= predator.reproduce_every:
                predator_kids += predator.kids
                mature_predators_count += 1

            color_red_sum_predator += color[0]
            color_green_sum_predator += color[1]
            color_blue_sum_predator += color[2]

            internal_angle = predator.current_angle
            internal_angle_radians = math.radians(internal_angle)
            rect = [center[0] - er,  center[1] - er, 2*er, 2*er]
            pygame.draw.rect(self.screen, color, rect)

            dx = 1.5 * er * math.cos(internal_angle_radians)
            dy = 1.5 * er * math.sin(internal_angle_radians)

            end_pos_x = round(center[0] + dx)
            end_pos_y = round(center[1] - dy)

            end_pos = (end_pos_x, end_pos_y)

            pygame.draw.line(self.screen, self.red, center, end_pos)

        avg_prey_kids = prey_kids / max(1, mature_preys_count)
        avg_prey_kids = round(avg_prey_kids, 2)
        text_preys = self.font.render('Prey Population: {}.       AK: {}'.format(n_preys, avg_prey_kids), True, self.black)
        text_preys_rect = text_preys.get_rect()
        text_preys_rect_x = self.default_size + (self.information_area // 2)
        text_preys_rect_y = self.text_start + (self.text_row_height)
        text_preys_rect.center = (text_preys_rect_x, text_preys_rect_y)

        avg_predator_kids = predator_kids / max(1, mature_predators_count)
        avg_predator_kids = round(avg_predator_kids, 2)
        text_predators = self.font.render('Predator Population: {}.    AK: {}'.format(n_predators, avg_predator_kids), True, self.black)
        text_predators_rect = text_predators.get_rect()
        text_predators_rect_x = self.default_size + (self.information_area // 2)
        text_predators_rect_y = self.text_start + (self.text_row_height * 2)
        text_predators_rect.center = (text_predators_rect_x, text_predators_rect_y)

        current_time = time.time()
        epoch_rate_per_epoch = round((current_time - self.start_time) / self.epoch, 2)
        epoch_rate = self.font.render('Time per epoch: {} seconds.'.format(epoch_rate_per_epoch), True, self.black)
        epoch_rate_rect = epoch_rate.get_rect()
        epoch_rate_rect_x = self.default_size + (self.information_area // 2)
        epoch_rate_rect_y = self.text_start + (self.text_row_height * 3)
        epoch_rate_rect.center = (epoch_rate_rect_x, epoch_rate_rect_y)

        average_age_prey = round(total_age_prey / max(n_preys, 1), 2)
        average_age_predator = round(total_age_predator / max(n_predators, 1), 2)
        text_age = self.font.render('Average Age- Prey: {}. Predator: {}.'.format(average_age_prey, average_age_predator),
                                    True, self.black)
        text_age_rect = text_age.get_rect()
        text_age_rect_x = self.default_size + (self.information_area // 2)
        text_age_rect_y = self.text_start + (self.text_row_height*4)
        text_age_rect.center = (text_age_rect_x, text_age_rect_y)

        self.screen.blit(text_preys, text_preys_rect)
        self.screen.blit(text_predators, text_predators_rect)
        self.screen.blit(epoch_rate, epoch_rate_rect)
        self.screen.blit(text_age, text_age_rect)

        if add_to_chart:
            self.color_red_average_prey.append(round(color_red_sum_prey / nm_prey, 2))
            self.color_green_average_prey.append(round(color_green_sum_prey / nm_prey, 2))
            self.color_blue_average_prey.append(round(color_blue_sum_prey / nm_prey, 2))

            self.color_red_average_predator.append(round(color_red_sum_predator / nm_predator, 2))
            self.color_green_average_predator.append(round(color_green_sum_predator / nm_predator, 2))
            self.color_blue_average_predator.append(round(color_blue_sum_predator / nm_predator, 2))
        self.ax2.clear()
        self.ax2.plot(range(len(self.color_red_average_prey)), self.color_red_average_prey, color='red',
                      label='Mobility')
        self.ax2.plot(range(len(self.color_green_average_prey)), self.color_green_average_prey, color='green',
                      label='Size')
        self.ax2.plot(range(len(self.color_blue_average_prey)), self.color_blue_average_prey, color='blue',
                      label='Visibility')
        # self.ax2.legend()
        self.ax3.clear()
        self.ax3.plot(range(len(self.color_red_average_predator)), self.color_red_average_predator, color='red',
                      label='Mobility')
        self.ax3.plot(range(len(self.color_green_average_predator)), self.color_green_average_predator, color='green',
                      label='Size')
        self.ax3.plot(range(len(self.color_blue_average_predator)), self.color_blue_average_predator, color='blue',
                      label='Visibility')
        # if self.display_legend:
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
            # self.display_legend = False
        self.generate_partition_line()

        chart_pos_x = self.default_size + 10
        chart_pos_y = 125

        # Convert to a Pygame image
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")

        self.screen.blit(surf, (chart_pos_x, chart_pos_y))

        pygame.display.flip()

