import pygame
import sys
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pygame.locals import *


class EvolutionUI:
    def __init__(self, screen_size: int, total_ticks: int):
        self.screen_size = screen_size
        self.screen = None
        self.entity_radius = 4
        # colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        self.information_area = 700
        self.font = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

        self.count_history_preys = []
        self.count_history_predators = []
        self.sum_history_entities = []

        self.color_red_average_prey = []
        self.color_green_average_prey = []
        self.color_blue_average_prey = []

        self.color_red_average_predator = []
        self.color_green_average_predator = []
        self.color_blue_average_predator = []

        self.history_append_probability = min(round((total_ticks ** 0.75) / total_ticks, 3) * 2, 1)
        self.epoch = 0

    def create_screen(self) -> pygame.Surface:
        pygame.init()
        screen = pygame.display.set_mode((self.screen_size + self.information_area, self.screen_size))
        pygame.display.set_caption('Evolution Simulation by blaze')
        self.screen = screen
        self.refresh_screen()

        font = pygame.font.Font('freesansbold.ttf', 32)
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

    def add_entities(self, preys, predators) -> None:

        # pygame stuff
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(1)

        # informatives

        text_epochs = self.font.render('Day: {}.'.format(self.epoch), True, self.black)
        self.epoch += 1
        text_epochs_rect = text_epochs.get_rect()
        text_epochs_rect_x = self.screen_size + (self.information_area // 2)
        text_epochs_rect_y = 40
        text_epochs_rect.center = (text_epochs_rect_x, text_epochs_rect_y)

        n_preys = len(preys)
        n_predators = len(predators)

        nm_prey = max(1, n_preys)
        nm_predator = max(1, n_predators)

        text_preys = self.font.render('Total Preys: {}.'.format(n_preys), True, self.black)
        text_preys_rect = text_preys.get_rect()
        text_preys_rect_x = self.screen_size + (self.information_area // 2)
        text_preys_rect_y = 80
        text_preys_rect.center = (text_preys_rect_x, text_preys_rect_y)

        text_predators = self.font.render('Total Predators: {}.'.format(n_predators), True, self.black)
        text_predators_rect = text_predators.get_rect()
        text_predators_rect_x = self.screen_size + (self.information_area // 2)
        text_predators_rect_y = 120
        text_predators_rect.center = (text_predators_rect_x, text_predators_rect_y)

        add_to_chart = False
        if np.random.choice([True, False], p=[self.history_append_probability, 1-self.history_append_probability]):
            add_to_chart = True
            self.count_history_preys.append(n_preys)
            self.count_history_predators.append(n_predators)
            self.sum_history_entities.append(n_preys + n_predators)

        self.screen.fill(self.white)


        # charts
        self.ax1.plot(range(len(self.count_history_preys)), self.count_history_preys, color='green')  # Example data
        self.ax1.plot(range(len(self.count_history_predators)), self.count_history_predators, color='red')
        self.ax1.set_title('Count of prey/predators')
        # self.ax1.set_xlabel('Data point')
        self.ax1.set_ylabel('No of entities')
        # self.ax2.plot(range(len(self.sum_history_entities)), self.sum_history_entities, color='blue')  # Example data
        # self.ax2.set_title('Count of total prey/predators')
        # self.ax2.set_xlabel('Data point')
        # self.ax2.set_ylabel('No fof entities')
        canvas = FigureCanvas(self.fig)
        canvas.draw()

        # Convert to a Pygame image
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")

        self.screen.blit(text_epochs, text_epochs_rect)
        self.screen.blit(text_preys, text_preys_rect)
        self.screen.blit(text_predators, text_predators_rect)

        color_red_sum_prey = 0
        color_green_sum_prey = 0
        color_blue_sum_prey = 0
        for prey in preys:
            center = prey.position
            color = prey.color
            er = prey.entity_radius

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

            pygame.draw.line(self.screen, self.black, center, end_pos)

        color_red_sum_predator = 0
        color_green_sum_predator = 0
        color_blue_sum_predator = 0
        for predator in predators:
            center = predator.position
            color = predator.color
            er = predator.entity_radius

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

            pygame.draw.line(self.screen, self.black, center, end_pos)



        if add_to_chart:
            self.color_red_average_prey.append(round(color_red_sum_prey / nm_prey, 2))
            self.color_green_average_prey.append(round(color_green_sum_prey / nm_prey, 2))
            self.color_blue_average_prey.append(round(color_blue_sum_prey / nm_prey, 2))

            self.color_red_average_predator.append(round(color_red_sum_predator / nm_predator, 2))
            self.color_green_average_predator.append(round(color_green_sum_predator / nm_predator, 2))
            self.color_blue_average_predator.append(round(color_blue_sum_predator / nm_predator, 2))

        self.ax2.plot(range(len(self.color_red_average_prey)), self.color_red_average_prey, color='red',
                      label='Mobility')
        self.ax2.plot(range(len(self.color_green_average_prey)), self.color_green_average_prey, color='green',
                      label='Size')
        self.ax2.plot(range(len(self.color_blue_average_prey)), self.color_blue_average_prey, color='blue',
                      label='Visibility')
        # self.ax2.legend()
        self.ax3.plot(range(len(self.color_red_average_predator)), self.color_red_average_predator, color='red',
                      label='Mobility')
        self.ax3.plot(range(len(self.color_green_average_predator)), self.color_green_average_predator, color='green',
                      label='Size')
        self.ax3.plot(range(len(self.color_blue_average_predator)), self.color_blue_average_predator, color='blue',
                      label='Visibility')
        # self.ax3.legend()
        self.generate_partition_line()

        chart_pos_x = self.screen_size - 10
        chart_pos_y = 150
        self.screen.blit(surf, (chart_pos_x, chart_pos_y))

        pygame.display.flip()
