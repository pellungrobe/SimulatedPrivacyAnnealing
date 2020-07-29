from datetime import datetime, timedelta


def weekend_weekdays(zero_at_day=0, number_of_days=31):
    day_dict = {1: "Monday", 2: "Tuesday", 3: "Wednesday",
                4: "Thursday", 5: "Friday", 6: "Saturday", 0: "Sunday"}
    day = 24
    day_num = zero_at_day
    weekdays, weekend = list(), list()
    for i in range(0, number_of_days+1):
        curr_day = day_dict[(day_num + i) % 7]
        for t in range(0, 24):
            timeslot = (i * 24) + t
            if curr_day in ["Sunday", "Saturday"]:
                weekend.append(timeslot)
            else:
                weekdays.append(timeslot)
    return weekend, weekdays


def morning_midday_afternoon_night(number_of_days=31):
    morning, midday, afternoon, night = list(), list(), list(), list()
    for i in range(0, number_of_days+1):
        for t in range(0, 24):
            timeslot = (i * 24) + t
            if t in range(0, 3) or t in range(21, 24):
                night.append(timeslot)
            elif t in range(3, 9):
                morning.append(timeslot)
            elif t in range(9, 15):
                midday.append(timeslot)
            else:
                afternoon.append(timeslot)
    return morning, midday, afternoon, night


def timeslot_partition(zero_at_day=0, number_of_days=31):
    weekend, weekdays = weekend_weekdays(zero_at_day, number_of_days)
    morning, midday, afternoon, night = morning_midday_afternoon_night(number_of_days)
    results = dict()
    for timeslot in range(0,(number_of_days*24)+1):
        if timeslot in morning:
            if timeslot in weekend:
                results[timeslot] = "morning_weekend"
            else:
                results[timeslot] = "morning_weekdays"
        elif timeslot in midday:
            if timeslot in weekend:
                results[timeslot] = "midday__weekend"
            else:
                results[timeslot] = "midday_weekdays"
        elif timeslot in afternoon:
            if timeslot in weekend:
                results[timeslot] = "afternoon__weekend"
            else:
                results[timeslot] = "afternoon_weekdays"
        else:
            if timeslot in weekend:
                results[timeslot] = "night__weekend"
            else:
                results[timeslot] = "night_weekdays"
    return results


def extract_all_times(time_string):
    year = int(time_string[:4])
    month = int(time_string[:6][-2:])
    day = int(time_string[:8][-2:])
    hour = int(time_string[:10][-2:])
    minute = int(time_string[:12][-2:])
    second = int(time_string[:14][-2:])
    return year, month, day, hour, minute, second


def add_0_before(str_time):
    if len(str_time) == 1:
        return '0' + str_time
    else:
        return str_time


def string_to_datetime(time_string):
    year, month, day, hour, minute, second = extract_all_times(time_string)
    return datetime(year, month, day, hour, minute, second)


def datetime_to_string(dt):
    str_second = add_0_before(str(dt.second))
    str_minute = add_0_before(str(dt.minute))
    str_hour = add_0_before(str(dt.hour))
    str_day = add_0_before(str(dt.day))
    str_month = add_0_before(str(dt.month))
    str_year = add_0_before(str(dt.year))
    return str_year + str_month + str_day + str_hour + str_minute + str_second


def int_to_datetime(time_int):
    return string_to_datetime(str(time_int))


def datetime_to_int(dt):
    return int(datetime_to_string(dt))


def nearest_timeslot(dt, timeslot_length):
    td = timedelta(seconds=timeslot_length)
    timeslot = datetime(dt.year, dt.month, 1, 0, 0, 0)
    while (dt > timeslot):
        timeslot += td
    return timeslot
