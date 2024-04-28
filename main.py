# imports
import ui
from entities import Prey, Predator
import helper
from sys import getsizeof

# EVOLUTION PARAMETERS
GRID_SIZE: int = 300
TOTAL_ENTITIES: int = 1000

PREY_LINE_OF_ACTIONS = 25
PREDATOR_LINE_OF_ACTIONS = 10

SUPPORTED_PREYS = 500  # means there is food for only 2000 preys

# Calculation parameters
TOTAL_PREYS: int = int(TOTAL_ENTITIES//1.5)
TOTAL_PREDATORS: int = (TOTAL_ENTITIES - TOTAL_PREYS)


TOTAL_TICKS = 5000


def main() -> None:
    world = ui.EvolutionUI(GRID_SIZE, TOTAL_TICKS)
    world_screen = world.create_screen()

    preys = []
    predators = []

    for i in range(TOTAL_PREYS):
        prey = Prey(GRID_SIZE)
        preys.append(prey)

    for i in range(TOTAL_PREDATORS):
        predator = Predator(GRID_SIZE)
        predators.append(predator)

    world.add_entities(preys, predators)

    for tick in range(1, TOTAL_TICKS+1):
        print("############### Tick: {} ###############".format(tick))

        ## find all entities which are in visible distance of each other
        prey_predator_visibility = {}
        predator_prey_visibility = {}
        n_preys = len(preys)
        n_predators = len(predators)
        for prey_i, prey in enumerate(preys):

            entity_visibility = prey.visibility
            entity_angle_of_vision = prey.angle_of_vision
            entity_current_angle = prey.current_angle

            # print("Entity current angle: {}".format(entity_current_angle))

            for predator_i, predator in enumerate(predators):
                prey_position = prey.position
                predator_position = predator.position

                prey_predator_distance = helper.calculate_distance(prey_position, predator_position)

                # check prey predator collision
                if prey_predator_distance <= entity_visibility:
                    # the enemy is in visibility range
                    # calculate angle coefficient
                    # calculate distance coefficient
                    # calculate which sector does the enemy belong

                    inscribed_angle = helper.calculate_angle(prey_position, predator_position)

                    apparent_inscribed_angle = inscribed_angle - entity_current_angle
                    if apparent_inscribed_angle < 0:
                        apparent_inscribed_angle = 360 + apparent_inscribed_angle

                    # print("Apparent inscribed angle: {}".format(apparent_inscribed_angle))

                    if helper.check_angle_in_visibility(entity_angle_of_vision, apparent_inscribed_angle,
                                                        entity_current_angle, inscribed_angle):

                        distance_coefficient = prey_predator_distance / entity_visibility
                        distance_coefficient = 1 - distance_coefficient
                        distance_coefficient = round(distance_coefficient, 3)

                        sector = helper.calculate_sector(inscribed_angle, entity_angle_of_vision,
                                                         PREY_LINE_OF_ACTIONS, entity_current_angle)

                        new_obj = {
                            "distance_coefficient": distance_coefficient,
                            "sector": sector,
                            "predator_i": predator_i
                        }
                        if prey_i not in prey_predator_visibility:
                            prey_predator_visibility[prey_i] = [new_obj]
                        else:
                            prey_predator_visibility[prey_i].append(new_obj)
                        # print(apparent_inscribed_angle, prey_predator_distance, sector)

                predator_visibility = predator.visibility
                predator_angle_of_vision = predator.angle_of_vision
                predator_current_angle = predator.current_angle
                # check predator prey collision
                if prey_predator_distance <= predator_visibility:
                    # the enemy is in visibility range
                    # calculate angle coefficient
                    # calculate distance coefficient
                    # calculate which sector does the enemy belong

                    inscribed_angle_predator = helper.calculate_angle(predator_position, prey_position)
                    apparent_inscribed_angle_predator = inscribed_angle_predator - predator_current_angle

                    if apparent_inscribed_angle_predator < 0:
                        apparent_inscribed_angle_predator = 360 + apparent_inscribed_angle_predator

                    if helper.check_angle_in_visibility(predator_angle_of_vision, apparent_inscribed_angle_predator,
                                                        predator_current_angle, inscribed_angle_predator):

                        distance_coefficient_predator = prey_predator_distance / predator_visibility
                        distance_coefficient_predator = 1 - distance_coefficient_predator
                        distance_coefficient_predator = round(distance_coefficient_predator, 3)

                        sector_predator = helper.calculate_sector(inscribed_angle_predator, predator_angle_of_vision,
                                                         PREDATOR_LINE_OF_ACTIONS, predator_current_angle)

                        new_obj = {
                            "distance_coefficient": distance_coefficient_predator,
                            "sector": sector_predator,
                            "prey_i": prey_i
                        }
                        if predator_i not in predator_prey_visibility:
                            predator_prey_visibility[predator_i] = [new_obj]
                        else:
                            predator_prey_visibility[predator_i].append(new_obj)
                        # print(inscribed_angle, prey_predator_distance, sector)


        # average_visibility_for_preys = sum([len(prey_predator_visibility[x]) for x in prey_predator_visibility]) / TOTAL_PREYS
        # average_visibility_for_predators = sum([len(predator_prey_visibility[x]) for x in predator_prey_visibility]) / TOTAL_PREDATORS
        #
        # print("Average visibility for preys: {}".format(average_visibility_for_preys))
        # print("Average visibility for predators: {}".format(average_visibility_for_predators))

        ## now we have all preys-predators that are in range of each other
        # for every prey predator call their brains to get the next move
        # lets start with preys
        prey_moves = {}
        for prey_i, prey in enumerate(preys):
            et_x, et_y = prey.position
            prey_neighbours = prey_predator_visibility.get(prey_i, [])
            if len(prey_neighbours) > 0:
                prey.set_opportunity()
            prey_inputs_n = PREY_LINE_OF_ACTIONS + 4
            # why the above thing?
            # the NN has 1 input for 1 line of action plus 4 inputs for distance from 4 edges
            prey_inputs = [0] * prey_inputs_n
            for neighbour in prey_neighbours:
                prey_inputs[neighbour['sector']] = neighbour['distance_coefficient']
            dx = et_x / GRID_SIZE
            dy = et_y / GRID_SIZE
            prey_inputs[-4] = dx
            prey_inputs[-3] = dy
            prey_inputs[-2] = 1 - dx
            prey_inputs[-1] = 1 - dy

            moves = prey.next_move(prey_inputs)
            prey_moves[prey_i] = moves

        predator_moves = {}
        for predator_i, predator in enumerate(predators):
            et_x, et_y = predator.position
            predator_neighbours = prey_predator_visibility.get(predator_i, [])
            if len(predator_neighbours) > 0:
                predator.set_opportunity()
            predator_inputs_n = PREDATOR_LINE_OF_ACTIONS + 4
            # why the above thing?
            # the NN has 1 input for 1 line of action plus 4 inputs for distance from 4 edges
            predator_inputs = [0] * predator_inputs_n

            for neighbour in predator_neighbours:
                # print("Predator neighbour sector: {}".format(neighbour['sector']))
                predator_inputs[neighbour['sector']] = neighbour['distance_coefficient']
            dx = et_x / GRID_SIZE
            dy = et_y / GRID_SIZE
            predator_inputs[-4] = dx
            predator_inputs[-3] = dy
            predator_inputs[-2] = 1 - dx
            predator_inputs[-1] = 1 - dy

            moves = predator.next_move(predator_inputs)
            predator_moves[predator_i] = moves

        # honour prey and predator moves
        for prey_i in prey_moves:
            prey = preys[prey_i]
            moves = prey_moves[prey_i]
            angle, distance = moves
            prey.move(angle, distance)

        for predator_i in predator_moves:
            predator = predators[predator_i]
            moves = predator_moves[predator_i]
            angle, distance = moves
            predator.move(angle, distance)
        print("{} MB used by Preys".format(helper.bytes_to_mb(getsizeof(preys))))
        print("{} MB used by Predators".format(helper.bytes_to_mb(getsizeof(predators))))
        world.add_entities(preys, predators)
        # time.sleep(0.1)

        # check for collisions
        delete_preys = []
        predators_that_got_food = []
        for prey_i, prey in enumerate(preys):
            for predator_i, predator in enumerate(predators):
                collision = helper.check_collision(prey, predator)
                if collision:
                    delete_preys.append(prey_i)
                    predators_that_got_food.append(predator_i)
                    break
        delete_preys = sorted(delete_preys, reverse=True)
        print("{} preys got eaten today".format(len(delete_preys)))
        print("{} predators got food today".format(len(predators_that_got_food)))
        for prey in delete_preys:
            # print(len(preys))
            if prey < len(preys):
                del preys[prey]
        # predators should eat and also give them feedback
        feedback_received = []
        for predator in predators_that_got_food:
            predators[predator].eat()
            if predator not in feedback_received:
                predators[predator].feedback()
                feedback_received.append(predator)
        # preys should gaze and also feedback as they survived
        for prey in preys:
            prey.gaze(n_preys, SUPPORTED_PREYS)
            prey.feedback()

        # give feedback to predators who had apportunity to eat but didnt
        for predator_i in range(len(predators)):
            if predator_i not in predators_that_got_food:
                if predators[predator_i].was_in_opportunity:
                    predators[predator_i].feedback(negative=True)

        ## delete preys and predators who have depleted energy
        delete_preys = []
        for prey_i, prey in enumerate(preys):
            if prey.check_if_about_to_die():
                delete_preys.append(prey_i)
        delete_preys = sorted(delete_preys, reverse=True)
        print("{} preys died today".format(len(delete_preys)))
        for prey in delete_preys:
            del preys[prey]

        delete_predators = []
        for predator_i, predator in enumerate(predators):
            if predator.check_if_about_to_die():
                delete_predators.append(predator_i)
        delete_predators = sorted(delete_predators, reverse=True)
        print("{} predators died today".format(len(delete_predators)))
        for predator in delete_predators:
            del predators[predator]


        ## check for entities who can reproduce
        new_preys = []
        new_predators = []

        for prey in preys:
            if prey.check_eligibility_for_reproduction():
                new_prey = prey.reproduce()
                new_preys.append(new_prey)
        print("{} preys born today".format(len(new_preys)))
        preys.extend(new_preys)

        for predator in predators:
            if predator.check_eligibility_for_reproduction():
                new_predator = predator.reproduce()
                new_predators.append(new_predator)
        print("{} predators born today".format(len(new_predators)))
        predators.extend(new_predators)

        # time.sleep(0.5)

main()

