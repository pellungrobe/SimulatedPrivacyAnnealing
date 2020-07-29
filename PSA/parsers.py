from .datetime_util import *
from operator import itemgetter
from .mobility_data import TrajectoryDataset, DatasetMaps


def read_geo_coord(filename):
    cell_coord = dict()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            itemlist = line.split(",")
            cell_index = int(itemlist[0])
            coord = (float(itemlist[1]), float(itemlist[2]))
            cell_coord[cell_index] = coord
    return cell_coord

def read_dataset_maps(filename, beginning_dt, ending_dt, timeslot_length):
    cell_coord = read_geo_coord(filename)
    return DatasetMaps(beginning_dt, ending_dt, timeslot_length, cell_coord)

def read_geo_coord_header(filename):
    cell_coord = dict()
    with open(filename) as f:
        next(f)
        for line in f:
            line = line.strip()
            itemlist = line.split(",")
            cell_index = int(itemlist[0])
            coord = (float(itemlist[1]), float(itemlist[2]))
            cell_coord[cell_index] = coord
    return cell_coord

def read_dataset_maps_header(filename, beginning_dt, ending_dt, timeslot_length):
    cell_coord = read_geo_coord_header(filename)
    return DatasetMaps(beginning_dt, ending_dt, timeslot_length, cell_coord)

def __read_presences(itemlist, timeslot_length):
    presences = {}
    for i in range(1, len(itemlist), 3):
        dt = string_to_datetime(itemlist[i + 2])
        timeslot = nearest_timeslot(dt, timeslot_length)
        if timeslot not in presences:
            presences[timeslot] = list()
        presences[timeslot].append(((float(itemlist[i]), float(itemlist[i + 1])), dt))
    return presences


def __longest_visits(presences):
    longest_visits = {}
    for timeslot in presences:
        sameslotlist = presences[timeslot]
        diffs = {}
        for i in range(0, len(sameslotlist)):
            loc_coord = sameslotlist[i][0]
            dt = sameslotlist[i][1]
            if i == len(sameslotlist) - 1:
                td = abs(dt - timeslot).total_seconds()
            else:
                td = abs(dt - sameslotlist[i + 1][1]).total_seconds()
            diffs[loc_coord] = td
        longest_visits[timeslot] = max(diffs.items(), key=itemgetter(1))[0]
    return longest_visits


def __read_trajectory_date_time_names(line, dataset_maps):
    line = line.strip()
    itemlist = line.split(",")
    triplet_list = list()
    presences = __read_presences(itemlist, dataset_maps.timeslot_length)
    visits = __longest_visits(presences)
    for timeslot in visits:
        triplet_list.append((itemlist[0], visits[timeslot], timeslot))
    return triplet_list


def read_trajectory_dataset_date_time_names(filename, dataset_maps):
    with open(filename) as f:
        triplets = list()
        for line in f:
            trplst = __read_trajectory_date_time_names(line, dataset_maps)
            for triplet in trplst:
                triplets.append(triplet)
    dataset = TrajectoryDataset(dataset_maps, triplets, true_names=True)
    return dataset


def write_trajectory_dataset_triplets(filename, dataset):
    with open(filename, "w+") as f:
        f.write(str(dataset))


def read_trajectory_dataset_triplets(filename, dataset_maps):
    triplets = list()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            itemlist = line.split(",")
            user_id = itemlist[0]
            cell_index = int(itemlist[1])
            timeslot_index = int(itemlist[2])
            triplets.append((user_id, cell_index, timeslot_index))
    dataset = TrajectoryDataset(dataset_maps, triplets)
    return dataset

def read_trajectory_dataset_triplets_header(filename, dataset_maps, compact=False):
    triplets = list()
    with open(filename) as f:
        next(f)
        for line in f:
            line = line.strip()
            itemlist = line.split(",")
            user_id = itemlist[0]
            cell_index = int(itemlist[1])
            timeslot_index = int(itemlist[2])
            triplets.append((user_id, cell_index, timeslot_index))
    dataset = TrajectoryDataset(dataset_maps, triplets, compact=compact)
    return dataset


def read_trajectory_dataset_synthetic(filename, dataset_maps):
    triplets = list()
    with open(filename) as f:
        next(f)
        for line in f:
            line = line.strip()
            itemlist = line.split(",")
            user_id = "synthetic_" + (itemlist[0])
            cell_index = int(itemlist[1])
            timeslot_index = int(itemlist[2])
            triplets.append((user_id, cell_index, timeslot_index))
    dataset = TrajectoryDataset(dataset_maps, triplets)
    return dataset


def write_colocations(colocations, filename):
    with open(filename, "w+") as f:
        for user in colocations:
            for user_coloc in colocations[user]:
                f.write(str(user) + "," + str(user_coloc) + "," + str(colocations[user][user_coloc]) + "\n")


def read_colocations(filename):
    result = dict()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            itemlist = line.split(",")
            if int(itemlist[0]) not in result:
                result[int(itemlist[0])] = {}
            result[int(itemlist[0])][int(itemlist[1])] = int(itemlist[2])
    return result


def read_attcker_risk(filename):
    risk_dict = dict()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            line = line.replace("\"", "")
            line = line.replace("}", "")
            line = line.replace("{", "")
            line = line.replace(" ", "")
            line = line.replace(":", ",")
            items = line.split(",")
            attacker = items[0]
            risks = list()
            for i in range(2, len(items), 2):
                risk = items[i]
                risks.append(float(risk))
            risk_dict[attacker] = risks
    return risk_dict


def total_coloc(colocations, user_id):
    count = 0
    for user_coloc in colocations[user_id]:
        count += colocations[user_id][user_coloc]
    return count


def colocation_distribution(colocations):
    result = []
    for user in colocations:
        result.append(total_coloc(colocations, user))
    return result


def write_sptial_tessellation(st, file):
    with open(file, "w+") as f:
        f.write("lat,lon,relevance\n")
        for i in st:
            f.write(str(st[i]["lat"]) + "," + str(st[i]["lon"]) + "," + str(st[i]["relevance"]) + "\n")
