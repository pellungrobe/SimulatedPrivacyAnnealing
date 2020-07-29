from PSA import anneal, datetime_util, matrix_moving_attack_annealer, mobility_data, parsers
from datetime import datetime
from numpy import mean

city_map #=spatial tessellation of the area, rows in the form (id of geographical cell, latitude, longitude)
init_datetime #=initial time of analysis in datetime
end_datetime #=ending time of analysis in datetime
t_len #= length of timeslot in seconds
trajectories #=trajectories in the form, (id of user,id of geographical cell, time_slot)
d_maps = parsers.read_dataset_maps_header(city_map, init_datetime,
                                              end_datetime, t_len)
dataset = parsers.read_trajectory_dataset_triplets_header(trajectories, d_maps)
mov_attack = matrix_moving_attack_annealer.MovingAttackOptimizer(dataset, initial_state_id=None, distance_limit=2.5, alpha=0.98,log_file=file_for_log)
solution, AAR = mov_attack.anneal()
