import math
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1024)
def calculate_distance(pos1: tuple, pos2: tuple) -> float:
    x_diff = (pos1[0] - pos2[0]) ** 2
    y_diff = (pos1[1] - pos2[1]) ** 2

    distance = math.sqrt(x_diff + y_diff)
    distance = round(distance, 2)
    return distance


@lru_cache(maxsize=1024)
def calculate_angle(pos1: tuple, pos2: tuple) -> float:
    x1, y1 = pos1
    x2, y2 = pos2

    dx = x2 - x1
    dy = y1 - y2

    angle_radians = math.atan2(dy, dx)
    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    # print("Original theta: {}".format(angle_degrees))
    if angle_degrees < 0:
        angle_degrees = 360 + angle_degrees

    return angle_degrees


@lru_cache(maxsize=1024)
def calculate_sector(inscribed_angle: float, angle_of_vision: float, line_of_attacks, entity_angle: float) -> int:
    angle_per_sector = angle_of_vision / line_of_attacks
    angle_diff = angle_difference(inscribed_angle, entity_angle)
    sector = int((angle_diff // angle_per_sector) + 1)
    # print("Inscribed angle: {}. Angle of vision: {}. Entity Angle: {}. Line of attacks: {}. Sector: {}".format(
    #     inscribed_angle, angle_of_vision, entity_angle, line_of_attacks, sector
    # ))
    return sector


def normalize_angle(angle):
    """ Normalize angle to be within 0 to 360 degrees """
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle


def is_angle_between(theta, alpha, beta):
    """ Check if angle theta is between alpha and beta anticlockwise """
    theta = normalize_angle(theta)
    alpha = normalize_angle(alpha)
    beta = normalize_angle(beta)

    if alpha < beta:
        return alpha < theta < beta
    else:
        return theta > alpha or theta < beta


def angle_difference(angle1, angle2):
    """ Calculate the smallest absolute difference between two angles """
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def check_angle_in_visibility(angle_of_visibility: float, angle: float, entity_angle: float, absolute_angle: float):
    angle_half = angle_of_visibility / 2

    return_item = False

    # check if angle is in between entity_angle -
    ll = entity_angle - angle_half
    ul = entity_angle + angle_half

    if is_angle_between(absolute_angle, ll, ul):
        return_item = True

    # print("Angle of visibility: {}. Angle: {}. Entity Angle: {}. Returned: {}".format(angle_of_visibility, angle,
    #                                                                                   entity_angle, return_item))
    return return_item


def check_collision(ent1, ent2):
    pos1 = ent1.position
    pos2 = ent2.position

    ent1_radius = ent1.entity_radius
    ent2_radius = ent2.entity_radius

    ent_distance = ent1_radius + ent2_radius
    distance = calculate_distance(pos1, pos2)

    if distance <= ent_distance:
        return True
    return False


def calculate_yes_no_probability(prob_for_true):
    return np.random.choice([True, False], p=[prob_for_true, 1 - prob_for_true])


def bytes_to_mb(bt):
    mb = bt / 1024  # kb
    mb = mb / 1024  # mb
    return round(mb, 2)


def get_normal_dist_random_number(mean, sigma):
    s = np.random.normal(0, 0.5, 1)
    return float(s[0])
