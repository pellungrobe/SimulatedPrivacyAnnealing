from random import random, randint, choice
from numpy import mean
from copy import deepcopy
from math import ceil, log10, exp
from json import dumps
from geopy.distance import geodesic
from operator import itemgetter
import sys
from time import time


def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - ceil(log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)


class MovingAttackOptimizer:
    start = None

    def __init__(self, dataset, distance_limit=None, initial_state_id=None, t_max=None, log_file=None, alpha=0.98, ones=None):
        self.alpha = alpha
        self.dataset = dataset
        self.risk = dict()
        self.risk_colocs = dict()
        self.distance_dict = dict()
        self.ones = ones
        for index in self.dataset.dataset_maps.coordinates:
            self.distance_dict[index] = list()
            if distance_limit is not None:
                for other_index in self.dataset.dataset_maps.coordinates:
                    dist = geodesic(self.dataset.dataset_maps.coordinates[index],
                                    self.dataset.dataset_maps.coordinates[other_index]).km
                    if dist <= distance_limit and index != other_index:
                        self.distance_dict[index].append(other_index)
            else:
                for other_index in self.dataset.dataset_maps.coordinates:
                    if index != other_index:
                        self.distance_dict[index].append(other_index)
        if initial_state_id is not None:
            trj_id_max = initial_state_id
        else:
            mean_risk = self.dataset.mean_risk_produced(list(self.dataset.trajectories.values()))
            trj_id_max = max(mean_risk.items(), key=itemgetter(1))[0]
        self.state = self.copy_state(self.dataset.trajectories[trj_id_max])
        self.state.id = "attacker" + str(self.state.id)
        self.starter_state = self.copy_state(self.state)
        self.steps = ceil(len(self.dataset.dataset_maps.coordinates) * len(self.dataset.dataset_maps.timeslots) * 0.03)
        self.__new_risk()
        self.best_energy = self.energy()
        self.best_state = self.copy_state(self.state)
        if log_file is not None:
            self.log_file = log_file
        else:
            self.log_file = "generic_log_file.txt"
        self.start = time()
        if t_max is not None:
            self.Tmax = t_max
        else:
            self.Tmax = self.find_max_T()

    def __new_risk(self):
        self.risk_colocs = dict()
        for i in range(0, len(self.state.visits)):
            individuals = self.dataset.total_matrix[self.state.visits[i], i]
            for ind in individuals:
                if ind in self.risk_colocs:
                    self.risk_colocs[ind].intersection(individuals)
                else:
                    self.risk_colocs[ind] = individuals
        for ind in self.dataset.trajectories:
            if ind in self.risk_colocs:
                self.risk[ind] = 1.0 / len(self.risk_colocs[ind])
            else:
                self.risk[ind] = 0

    def __update_risk(self, previous_index, new_index):
        previous_individuals = self.dataset.total_matrix[previous_index[0], previous_index[1]]
        new_individuals = self.dataset.total_matrix[new_index[0], new_index[1]]
        to_update = previous_individuals.union(new_individuals)
        for ind in to_update:
            if ind in self.risk_colocs:
                self.risk_colocs.pop(ind)
        for timeslot in range(0, len(self.state.visits)):
            individuals = self.dataset.total_matrix[self.state.visits[timeslot], timeslot]
            found = individuals.intersection(to_update)
            for ind_found in found:
                if ind_found in self.risk_colocs:
                    self.risk_colocs[ind_found].intersection(individuals)
                else:
                    self.risk_colocs[ind_found] = individuals
        for ind in to_update:
            if ind in self.risk_colocs:
                self.risk[ind] = 1.0 / len(self.risk_colocs[ind])
            else:
                self.risk[ind] = 0

    def random_generator(self):
        chosen_timeslot = randint(0, len(self.state.visits) - 1)
        chosen_location = self.state.visits[chosen_timeslot]
        if self.distance_dict[chosen_location]:
            new_location = choice(self.distance_dict[chosen_location])
        else:
            new_location = chosen_location
        self.state.visits[chosen_timeslot] = new_location
        self.__update_risk((chosen_location, chosen_timeslot), (new_location, chosen_timeslot))

    def move(self):
        self.random_generator()
        
    def mymean(self):
        return 1.0 - mean(list(self.risk.values()))
    
    def myones(self):
        return 1.0 - (list(self.risk.values()).count(1)/len(self.risk))

    def energy(self):
        if self.ones != None:
            return self.myones()
        else:
            return self.mymean()

    def copy_state(self, state):
        return deepcopy(state)

    def update(self, *args, **kwargs):
        self.update_dict(*args, **kwargs)

    def update_dict(self, step, temp, energy, acceptance, improvement):
        up_d = {"steps": step, "temperature": temp, "energy": energy, "acceptance_rate": acceptance,
                "improvement_rate": improvement, "best_energy": self.best_energy, "elapsed_time": time() - self.start}
        print(up_d)
        with open(self.log_file, "a+") as f:
            f.write(dumps(up_d))
            f.write("\n")
        return up_d

    def default_update(self, step, temp, energy, acceptance, improvement):
        elapsed = time() - self.start
        if step == 0:
            print(' Temperature        Energy    Accept   Improve     Elapsed',
                  file=sys.stderr)
            print('\r%12.5f  %12.2f                      %s' %
                  (temp, energy, time_string(elapsed)), file=sys.stderr, end="\r")
            sys.stderr.flush()
        else:
            print('\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s\r' %
                  (temp, energy, 100.0 * acceptance, 100.0 * improvement,
                   time_string(elapsed)), file=sys.stderr, end="\r")
            sys.stderr.flush()

    def cooling_schedule(self, *args, **kwargs):
        return self.exponential_decrese(*args, **kwargs)

    def exponential_decrese(self, temp):
        return temp * self.alpha

    def run(self, temp):
        """Anneals a system at constant temperature and returns the state,
        energy, rate of acceptance, and rate of improvement."""
        energy = self.energy()
        prev_state = self.copy_state(self.state)
        prev_energy = energy
        accepts, improves = 0, 0
        for _ in range(self.steps):
            self.move()
            energy = self.energy()
            energy_delta = energy - prev_energy
            if energy_delta > 0.0 and exp(-energy_delta / temp) < random():
                self.state = self.copy_state(prev_state)
                energy = prev_energy
            else:
                accepts += 1
                if energy_delta < 0.0:
                    improves += 1
                prev_state = self.copy_state(self.state)
                prev_energy = energy
                if energy < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = energy
        return energy, float(accepts) / self.steps, float(improves) / self.steps

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        # Note initial state
        temp = self.Tmax
        accepts, improves = 1, 1
        while accepts > 0.025 or improves > 0.005:
            energy, accepts, improves = self.run(temp)
            step += self.steps
            self.update(step, temp, energy, accepts, improves)
            temp = self.cooling_schedule(temp)

        self.state = self.copy_state(self.best_state)
        # Return best state and energy
        return self.best_state, self.best_energy

    def find_max_T(self):
        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        step = 0
        temp = 0.0
        energy = self.energy()
        self.update(step, temp, energy, None, None)
        while temp == 0.0:
            step += 1
            self.move()
            temp = abs(self.energy() - energy)

        # Search for Tmax - a temperature that gives 98% acceptance
        energy, acceptance, improvement = self.run(temp)

        step += self.steps
        while acceptance > 0.98:
            temp = round_figures(temp / 1.5, 2)
            energy, acceptance, improvement = self.run(temp)
            step += self.steps
            self.update(step, temp, energy, acceptance, improvement)
        while acceptance < 0.98:
            temp = round_figures(temp * 1.5, 2)
            energy, acceptance, improvement = self.run(temp)
            step += self.steps
            self.update(step, temp, energy, acceptance, improvement)
        return (temp)
