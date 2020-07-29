from bidict import bidict
from datetime import timedelta
from .datetime_util import timeslot_partition, weekend_weekdays
from numpy import zeros, mean
from geopy.distance import geodesic
from math import log
from statistics import stdev


class DatasetMaps:
    """To represent the maps of a dataset: in terms of cell names, centroids coordinates and timeslots"""

    def __init__(self, beginning_dt, end_dt, increment, cell_coord):
        """mapping coordinates and cellnames, generating timeslots"""
        self.coordinates = bidict()
        self.compact_coordinates = bidict()
        self.timeslots = bidict()
        self.timeslot_length = increment
        for cell_index in cell_coord:
            coord = cell_coord[cell_index]
            self.coordinates[cell_index] = coord
        i = 0
        for cell_index in self.coordinates:
            self.compact_coordinates[i] = self.coordinates[cell_index]
            i+=1
        count = 0
        td = timedelta(seconds=self.timeslot_length)
        last_generated = beginning_dt
        while last_generated <= end_dt:
            self.timeslots[count] = last_generated
            last_generated = last_generated + td
            count += 1

    def reverse_slot(self, cell_index, time_index):
        """Accesses bidictionaries with indexes, returning actual cell name and timeslot"""
        coord = self.coordinates[cell_index]
        time = self.timeslots[time_index]
        return coord, time

    def translate_slot(self, coord, datetime):
        """Accesses bidictionaries with cellname and timeslot to obtain matrix index"""
        cell_index = self.coordinates.inv[coord]
        time_index = self.timeslots.inv[datetime]
        return cell_index, time_index
    
    def reverse_slot_compact(self, cell_index, time_index):
        """Accesses bidictionaries with indexes, returning actual cell name and timeslot"""
        coord = self.compact_coordinates[cell_index]
        time = self.timeslots[time_index]
        return coord, time

    def translate_slot_compact(self, coord, datetime):
        """Accesses bidictionaries with cellname and timeslot to obtain matrix index"""
        cell_index = self.compact_coordinates.inv[coord]
        time_index = self.timeslots.inv[datetime]
        return cell_index, time_index
    
    def translate_cell_compact(self, cell_index):
        coord = self.coordinates[cell_index]
        return self.compact_coordinates.inv[coord]


class _Trajectory:

    def __init__(self, individual_id, number_of_timelosts):
        self.visits = [-1] * number_of_timelosts
        self.id = individual_id

    def __repr__(self):
        return str(self.visits)

    def __deepcopy__(self, memo=dict()):
        new_one = _Trajectory(self.id, len(self.visits))
        for i in range(0, len(self.visits)):
            new_one.visits[i] = self.visits[i]
        return new_one


class TrajectoryDataset:

    def __init__(self, dataset_maps, visit_list, true_names=False, compact = False):
        self.dataset_maps = dataset_maps
        self.trajectories = dict()
        for triplet in visit_list:
            self.__add_trajectory(triplet[0])
            self.__add_visit(triplet[1], triplet[2], triplet[0], true_names)
        self.total_matrix = zeros((max(self.dataset_maps.coordinates.keys())+1, len(self.dataset_maps.timeslots)), dtype=object)
        for i in range(0, self.total_matrix.shape[0]):
            for j in range(0, self.total_matrix.shape[1]):
                self.total_matrix[i, j] = set()
        self.__complete_dataset_structures()
        if compact:
            self.total_matrix = zeros((max(self.dataset_maps.compact_coordinates.keys())+1, len(self.dataset_maps.timeslots)), dtype=object)
            for i in range(0, self.total_matrix.shape[0]):
                for j in range(0, self.total_matrix.shape[1]):
                    self.total_matrix[i, j] = set()
            for trj_id in self.trajectories:
                trj = self.trajectories[trj_id]
                for i in range(0,len(trj.visits)):
                    comp = dataset_maps.translate_cell_compact(trj.visits[i])
                    trj.visits[i] = comp
            self.__complete_dataset_structures()

    def __complete_total_matrix(self):
        for i in range(0, self.total_matrix.shape[0]):
            for j in range(0, self.total_matrix.shape[1]):
                self.total_matrix[i, j] = set()
        for trj_id in self.trajectories:
            trj = self.trajectories[trj_id]
            for timeslot in range(0, len(trj.visits)):
                self.total_matrix[trj.visits[timeslot], timeslot].add(trj_id)

    def __autocomplete_trajectory(self, trj_id):
        trj = self.trajectories[trj_id]
        started = False
        for timeslot in range(0, len(trj.visits)):
            if trj.visits[timeslot] == -1 and started:
                trj.visits[timeslot] = trj.visits[timeslot - 1]
            if trj.visits[timeslot] != -1 and not started:
                started = True
                for n in range(0, timeslot):
                    trj.visits[n] = trj.visits[timeslot]

    def __autocomplete_all_trajectories(self):
        for trj_id in self.trajectories:
            self.__autocomplete_trajectory(trj_id)

    def __complete_dataset_structures(self):
        self.__autocomplete_all_trajectories()
        self.cut_trajectories(self.filter_on_trips_general())

    def __add_trajectory(self, individual_id):
        if individual_id not in self.trajectories:
            trj = _Trajectory(individual_id, len(self.dataset_maps.timeslots))
            self.trajectories[individual_id] = trj

    def __add_visit(self, cell, time, trj_id, true_names=False, compact=False):
        trj = self.trajectories[trj_id]
        if true_names:
            if compact:
                cell_index, time_index = self.dataset_maps.translate_slot_compact(cell, time)
            else:
                cell_index, time_index = self.dataset_maps.translate_slot(cell, time)
            trj.visits[time_index] = cell_index
        else:
            trj.visits[time] = cell

    def statistics(self):
        stats = ["unique_locations", "number_of_trips", "entropy", "max_distance", "mean_distance", "radius"]
        statistics = {k:list() for k in stats}
        for trj_id in self.trajectories:
            statistics["unique_locations"].append(len(self.unique_locations(trj_id)))
            statistics["number_of_trips"].append(len(self.trips(trj_id)))
            distances = self.distances_traveled(trj_id)
            if distances:
                statistics["max_distance"].append(max(distances))
                statistics["mean_distance"].append(mean(distances))
            else:
                statistics["max_distance"].append(0)
                statistics["mean_distance"].append(0)
            statistics["entropy"].append(self.entropy(trj_id))
            statistics["radius"].append(self.radius_of_gyration(trj_id))
        return statistics

    def unique_locations(self, trj_id):
        trj = self.trajectories[trj_id]
        uniques = set()
        for cell in trj.visits:
            uniques.add(cell)
        return uniques

    def number_of_visits(self, trj_id, cell_index):
        trj = self.trajectories[trj_id]
        return trj.visits.count(cell_index)

    def visits_of_location(self, cell_index):
        count = 0
        for trj_id in self.trajectories:
            trj = self.trajectories[trj_id]
            count += self.number_of_visits(trj, cell_index)
        return count

    def trips(self, trj_id):
        trj = self.trajectories[trj_id]
        prev_loc = trj.visits[0]
        trips = list()
        trips.append((prev_loc,0))
        for timeslot in range(1, len(trj.visits)):
            current_loc = trj.visits[timeslot]
            if current_loc != prev_loc:
                trips.append((current_loc, timeslot))
                prev_loc = current_loc
        return trips

    def filter_on_trips_periodically(self, trip_limit):
        to_cut = list()
        for trj_id in self.trajectories:
            trips = self.trips(trj_id)
            current_trips = 0
            current_day = 1
            for pair in trips:
                timeslot = self.dataset_maps.timeslots[pair[1]]
                day = timeslot.day
                if day == current_day:
                    current_trips += 1
                else:
                    if day - current_day > 1:
                        to_cut.append(trj_id)
                        break
                    if current_trips < trip_limit:
                        to_cut.append(trj_id)
                        break
                    else:
                        current_day+=1
                        current_trips = 1
        return to_cut

    def filter_on_trips_general(self, trip_limit=2):
        to_cut = list()
        for trj_id in self.trajectories:
            trips = self.trips(trj_id)
            if len(trips) < trip_limit:
                to_cut.append(trj_id)
        return to_cut

    def cut_trajectories(self, id_list_to_cut):
        for trj_id in id_list_to_cut:
            self.trajectories.pop(trj_id)
        self.__complete_total_matrix()

    def distances_traveled(self, trj_id, compact = False):
        trips = self.trips(trj_id)
        distances = list()
        if trips:
            prev_loc = trips[0][0]
            for pair_index in range(1,len(trips)):
                pair = trips[pair_index]
                loc = pair[0]
                if compact:
                    distance = geodesic(self.dataset_maps.compact_coordinates[prev_loc], self.dataset_maps.compact_coordinates[loc]).km
                else:
                    distance = geodesic(self.dataset_maps.coordinates[prev_loc], self.dataset_maps.coordinates[loc]).km
                distances.append(distance)
                prev_loc = loc
        return distances

    def trips_distribution(self, trj_id, starting_day=0, num_days=31,tp=None):
        trips = self.trips(trj_id)
        if tp is None:
            time_part = timeslot_partition(starting_day,num_days)
        else:
            time_part = tp
        results = {k:0 for k in ["morning_weekend","morning_weekdays","midday__weekend","midday_weekdays","afternoon__weekend",
                      "afternoon_weekdays","night__weekend","night_weekdays"]}
        for pair in trips:
            timeslot = pair[1]
            partition = time_part[timeslot]
            results[partition] += 1
        return results

    def trips_distribution_hourly(self, starting_day=0, num_days=31):
        weekend, weekdays = weekend_weekdays()
        results={"weekend":{k:0 for k in range(0,24)}, "weekdays":{k:0 for k in range(0,24)}}
        norm = 0
        for trj_id in self.trajectories:
            trips = self.trips(trj_id)
            norm += len(trips)
            for pair in trips:
                timeslot = pair[1]
                t = timeslot%24
                if timeslot in weekend:
                    results["weekend"][t] += 1
                else:
                    results["weekdays"][t] += 1
        for timeslot in results["weekend"]:
            results["weekend"][timeslot] = results["weekend"][timeslot]/norm
        for timeslot in results["weekdays"]:
            results["weekdays"][timeslot] = results["weekdays"][timeslot]/norm
        return results

    def trips_distribution_all(self,starting_day=0, num_days=31):
        results = {k: 0 for k in
                   ["morning_weekend", "morning_weekdays", "midday__weekend", "midday_weekdays", "afternoon__weekend",
                    "afternoon_weekdays", "night__weekend", "night_weekdays"]}
        for trj_id in self.trajectories:
            time_part = timeslot_partition(starting_day,num_days)
            trips_distribution = self.trips_distribution(trj_id,starting_day, num_days, time_part)
            for k in trips_distribution:
                results[k] += trips_distribution[k]
        return results

    def location_rank(self, trj_id):
        trj = self.trajectories[trj_id]
        freqs  = dict()
        for cell in trj.visits:
            if cell not in freqs:
                freqs[cell] = 1
            else:
                freqs[cell] += 1
        return freqs, {key: rank for rank, key in enumerate(sorted(freqs, key=freqs.get, reverse=True),1)}

    def entropy(self, trj_id):
        trj = self.trajectories[trj_id]
        cell_freq = dict()
        tot_freq = 0.0
        for cell_index in trj.visits:
            if cell_index not in cell_freq:
                cell_freq[cell_index] = 1
            else:
                cell_freq[cell_index] += 1
            tot_freq += 1
        cell_prob = {k:float(v)/tot_freq for (k,v) in cell_freq.items()}
        entropy = 0
        for v in cell_prob.values():
            entropy += v*log(v,2)
        return -entropy

    def mean_entropy(self):
        entr = list()
        for trj_id in self.trajectories:
            entr.append(self.entropy(trj_id))
        return sum(entr) / len(entr)

    def radius_of_gyration(self, trj_id):
        trj = self.trajectories[trj_id]
        cell_frequency = dict()
        sum_for_center = [0, 0]
        n = 0
        for timeslot in range(0, len(trj.visits)):
            cell = trj.visits[timeslot]
            if cell in cell_frequency:
                cell_frequency[cell] += 1
            else:
                cell_frequency[cell] = 1
            sum_for_center[0] += self.dataset_maps.coordinates[cell][0]
            sum_for_center[1] += self.dataset_maps.coordinates[cell][1]
            n += 1
        terms = list()
        center_of_mass = (sum_for_center[0] / n, sum_for_center[1] / n)
        for cell in cell_frequency:
            cell_freq = cell_frequency[cell]
            distance = geodesic(self.dataset_maps.coordinates[cell], center_of_mass)
            terms.append(cell_freq * (distance.km ** 2))
        summation = sum(terms) / n
        result = summation ** 0.5
        return result

    def mean_radius(self):
        rads = list()
        for trj_id in self.trajectories:
            rads.append(self.radius_of_gyration(trj_id))
        return sum(rads) / len(rads)

    def fit_geo_maps(self):
        new_maps_coordinates = bidict()
        for trj_id in self.trajectories:
            trj = self.trajectories[trj_id]
            for cell_index in trj.visits:
                if cell_index not in new_maps_coordinates:
                    new_maps_coordinates[cell_index] = self.dataset_maps.coordinates[cell_index]
        self.dataset_maps.coordinates = new_maps_coordinates

    def risk_produced(self, trj_list):
        results = dict()
        for trj in trj_list:
            results[trj.id] = dict()
        for trj in trj_list:
            risk_colocs = dict()
            risk_effective = dict()
            for time_slot in range(0, len(trj.visits)):
                cell = trj.visits[time_slot]
                individuals = self.total_matrix[cell, time_slot]
                for ind in individuals:
                    if ind in risk_colocs:
                        risk_colocs[ind].intersection(individuals)
                    else:
                        risk_colocs[ind] = individuals
            for ind in self.trajectories:
                if ind not in risk_colocs:
                    risk_effective[ind] = 0
                else:
                    risk_effective[ind] = 1.0 / len(risk_colocs[ind])
            results[trj.id] = risk_effective
        return results
    
    def risk_produced_list(self, visit_list):
        results = list()
        risk_colocs = dict()
        risk_effective = list()
        for time_slot in range(0, len(visit_list)):
            cell = visit_list[time_slot]
            individuals = self.total_matrix[cell, time_slot]
            for ind in individuals:
                if ind in risk_colocs:
                    risk_colocs[ind].intersection(individuals)
                else:
                    risk_colocs[ind] = individuals
        for ind in self.trajectories:
            if ind not in risk_colocs:
                risk_effective.append(0)
            else:
                risk_effective.append(1.0 / len(risk_colocs[ind]))
        results.append(risk_effective)
        return results

    def risk_analysis(self, trj_visits):
        mean_risk_in_time = list()
        reid_cumulative = list()
        risk_colocs = dict()
        risk_effective = dict()
        risk_dev = list()
        for ind in self.trajectories:
            risk_effective[ind] = 0
        for time_slot in range(0, len(trj_visits)):
            reid_cumulative.append(0)
        for time_slot in range(0, len(trj_visits)):
            cell = trj_visits[time_slot]
            individuals = self.total_matrix[cell, time_slot]
            for ind in individuals:
                if ind in risk_colocs:
                    risk_colocs[ind].intersection(individuals)
                else:
                    risk_colocs[ind] = individuals
                risk_effective[ind] = 1.0 / len(risk_colocs[ind])
                if risk_effective[ind] == 1:
                    reid_cumulative[time_slot] += 1
            mean_risk_in_time.append(mean(list(risk_effective.values())))
            risk_dev.append(stdev(list(risk_effective.values())))
        for i in range(1,len(reid_cumulative)):
            reid_cumulative[i] += reid_cumulative[i-1]
        return mean_risk_in_time, reid_cumulative, risk_effective, risk_dev

    def mean_risk_produced(self, trj_list):
        risk_dict = self.risk_produced(trj_list)
        print("fatt0")
        risk_mean = {k: mean(list(v.values())) for (k, v) in risk_dict.items()}
        return risk_mean

    def origin_destination_matrix(self):
        odm = zeros((len(self.dataset_maps.compact_coordinates), len(self.dataset_maps.compact_coordinates)), dtype=int)
        for ind in self.trajectories:
            trj = self.trajectories[ind]
            prev_cell = trj.visits[0]
            for timeslot in trj.visits:
                current_cell = trj.visits[timeslot]
                if current_cell != prev_cell:
                    odm[prev_cell, current_cell] += 1
                    prev_cell = current_cell
        return odm

    def spatial_tessellation(self):
        st = dict()
        for cell_index in self.dataset_maps.coordinates:
            relevance = 0
            for timeslot in range(0, len(self.dataset_maps.timeslots)):
                relevance += len(self.total_matrix[cell_index][timeslot])
            st[cell_index] = {'lat': self.dataset_maps.coordinates[cell_index][0],
                              'lon': self.dataset_maps.coordinates[cell_index][1],
                              'relevance': relevance}
        return st

    def colocations(self, trj_1, trj_2):
        result = list()
        for timeslot in range(0, len(trj_1.visits)):
            if trj_1.visits[timeslot] == trj_2.visits[timeslot]:
                result.append((trj_1.visits[timeslot], timeslot))
        return result

    def total_colocations(self):
        colocs = dict()
        rows, columns = self.total_matrix.nonzero()
        count = 0
        for cell, timeslot in zip(rows, columns):
            for x in range(0, len(self.total_matrix[cell, timeslot])):
                individual1 = self.total_matrix[cell, timeslot][x]
                if individual1 not in colocs:
                    colocs[individual1] = dict()
                for y in range(0, len(self.total_matrix[cell, timeslot])):
                    individual2 = self.total_matrix[cell, timeslot][y]
                    if individual1 != individual2:
                        if individual2 not in colocs[individual1]:
                            colocs[individual1][individual2] = dict()
                            colocs[individual1][individual2] = 1
                        else:
                            colocs[individual1][individual2] += 1
            count += 1
            print("Complete " + str(count / len(rows)))
        for ind in self.trajectories:
            if ind not in colocs:
                colocs[ind] = dict()
            for other in self.trajectories:
                if other != ind and other not in colocs[ind]:
                    colocs[ind][other] = 0
        return colocs

    def __repr__(self):
        res = ""
        for trj_id in self.trajectories:
            res += self.string_repr(trj_id)
        return res

    def string_repr(self, trj_id, true_names=False, compact=False):
        trj = self.trajectories[trj_id]
        result = ""
        for timeslot in range(0, len(trj.visits)):
            if true_names:
                if compact:
                    cell, time = self.dataset_maps.reverse_slot_compact(trj.visits[timeslot], timeslot)
                else:
                    cell, time = self.dataset_maps.reverse_slot(trj.visits[timeslot], timeslot)
                result += str(trj.id) + "," + str(cell[0]) + "," + str(cell[1]) + "," + str(time) + "\n"
            else:
                result += str(trj.id) + "," + str(trj.visits[timeslot]) + "," + str(timeslot) + "\n"
        return result
