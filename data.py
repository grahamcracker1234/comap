import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./data.csv")

# Columns
# 1 --> Date | 2 --> Day of the week | 3 --> Contest Number | 4 --> Word | 5 --> Results Count | 6 --> Hard Mode Count
# 7 --> First Try | 8 --> 2 Tries | 9 --> 3 Tries | 10 --> 4 Tries | 11 --> 5 Tries | 12 --> 6 Tries | 13 --> Failed 

# January 7th was a Friday

# Make it into an array as I am not a panda guy :D
array_data = df.to_numpy()


def participationOverTime():
    """
    This function will graph participation over time
    """
    # First thing, obtain a list with participation per day and graph it
    participation_count = []
    for entry in array_data:
        participation_count.append(entry[4])

    num_days = list(range(1, len(array_data)+1))

    plt.plot(num_days,participation_count)
    plt.xlabel("Days")
    plt.ylabel("Participation Count")
    plt.title("Participation Count over time")
    plt.show()

def participationHardModeOverTime():
    """
    This function will graph participation over time
    """
    # First thing, obtain a list with participation per day and graph it
    participation_count = []
    for entry in array_data:
        participation_count.append(entry[5])

    num_days = list(range(1, len(array_data)+1))

    plt.plot(num_days,participation_count)
    plt.xlabel("Days")
    plt.ylabel("Hard Mode Participation Count")
    plt.title("Hard Mode Participation Count over time")
    plt.show()

def participationPerDay():
    """
    This function will graph participation per day. Bar Graph makes sense as it is Categorical data
    """
    # First day is a friday
    participation_per_day = {"Monday": [], "Tuesday": [], "Wednesday": [], "Thursday": [], "Friday": [], "Saturday": [], "Sunday": []}
    for e in array_data:
        if e[1] == "Friday": participation_per_day["Friday"].append(e[4])
        elif e[1] == "Saturday": participation_per_day["Saturday"].append(e[4])
        elif e[1] == "Sunday": participation_per_day["Sunday"].append(e[4])
        elif e[1] == "Monday": participation_per_day["Monday"].append(e[4])
        elif e[1] == "Tuesday": participation_per_day["Tuesday"].append(e[4])
        elif e[1] == "Wednesday": participation_per_day["Wednesday"].append(e[4])
        elif e[1] == "Thursday": participation_per_day["Thursday"].append(e[4])

    # plot Graph for average participation per day
    avg_participation_per_day = []
    avg_participation_per_day.append(round(np.average(participation_per_day["Monday"]),2))
    avg_participation_per_day.append(round(np.average(participation_per_day["Tuesday"]),2))
    avg_participation_per_day.append(round(np.average(participation_per_day["Wednesday"]),2))
    avg_participation_per_day.append(round(np.average(participation_per_day["Thursday"]),2))
    avg_participation_per_day.append(round(np.average(participation_per_day["Friday"]),2))
    avg_participation_per_day.append(round(np.average(participation_per_day["Saturday"]),2))
    avg_participation_per_day.append(round(np.average(participation_per_day["Sunday"]),2))

    plt.bar(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],avg_participation_per_day)
    plt.xlabel("Days")
    plt.ylabel("Average Participation Count")
    plt.title("Average Participation Count per day")
    plt.show()

def successRatesPerDay():
    """
    This function will graph the number of success rates per day
    """
    participation_per_day = {"Monday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0},
                             "Tuesday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0}, 
                             "Wednesday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0},
                             "Thursday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0},
                             "Friday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0},
                             "Saturday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0},
                             "Sunday": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "X": 0}}

    for e in array_data:
        if e[1] == "Friday":
            # Friday
            participation_per_day["Friday"]["1"] += e[6]
            participation_per_day["Friday"]["2"] += e[7]
            participation_per_day["Friday"]["3"] += e[8]
            participation_per_day["Friday"]["4"] += e[9]
            participation_per_day["Friday"]["5"] += e[10]
            participation_per_day["Friday"]["6"] += e[11]
            participation_per_day["Friday"]["X"] += e[12]
        elif e[1] == "Saturday":
            # Saturday
            participation_per_day["Saturday"]["1"] += e[6]
            participation_per_day["Saturday"]["2"] += e[7]
            participation_per_day["Saturday"]["3"] += e[8]
            participation_per_day["Saturday"]["4"] += e[9]
            participation_per_day["Saturday"]["5"] += e[10]
            participation_per_day["Saturday"]["6"] += e[11]
            participation_per_day["Saturday"]["X"] += e[12]
        elif e[1] == "Sunday":
            # Sunday
            participation_per_day["Sunday"]["1"] += e[6]
            participation_per_day["Sunday"]["2"] += e[7]
            participation_per_day["Sunday"]["3"] += e[8]
            participation_per_day["Sunday"]["4"] += e[9]
            participation_per_day["Sunday"]["5"] += e[10]
            participation_per_day["Sunday"]["6"] += e[11]
            participation_per_day["Sunday"]["X"] += e[12]
        elif e[1] == "Monday":
            # Monday
            participation_per_day["Monday"]["1"] += e[6]
            participation_per_day["Monday"]["2"] += e[7]
            participation_per_day["Monday"]["3"] += e[8]
            participation_per_day["Monday"]["4"] += e[9]
            participation_per_day["Monday"]["5"] += e[10]
            participation_per_day["Monday"]["6"] += e[11]
            participation_per_day["Monday"]["X"] += e[12]
        elif e[1] == "Tuesday":
            # Tuesday
            participation_per_day["Tuesday"]["1"] += e[6]
            participation_per_day["Tuesday"]["2"] += e[7]
            participation_per_day["Tuesday"]["3"] += e[8]
            participation_per_day["Tuesday"]["4"] += e[9]
            participation_per_day["Tuesday"]["5"] += e[10]
            participation_per_day["Tuesday"]["6"] += e[11]
            participation_per_day["Tuesday"]["X"] += e[12]
        elif e[1] == "Wednesday":
            # Wednesday
            participation_per_day["Wednesday"]["1"] += e[6]
            participation_per_day["Wednesday"]["2"] += e[7]
            participation_per_day["Wednesday"]["3"] += e[8]
            participation_per_day["Wednesday"]["4"] += e[9]
            participation_per_day["Wednesday"]["5"] += e[10]
            participation_per_day["Wednesday"]["6"] += e[11]
            participation_per_day["Wednesday"]["X"] += e[12]
        elif e[1] == "Thursday":
            # Thursday
            participation_per_day["Thursday"]["1"] += e[6]
            participation_per_day["Thursday"]["2"] += e[7]
            participation_per_day["Thursday"]["3"] += e[8]
            participation_per_day["Thursday"]["4"] += e[9]
            participation_per_day["Thursday"]["5"] += e[10]
            participation_per_day["Thursday"]["6"] += e[11]
            participation_per_day["Thursday"]["X"] += e[12]

    N = 7
    ind = np.arange(N)  # the x locations for the groups
    width = 0.1     # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    first_tries = [participation_per_day["Monday"]["1"], participation_per_day["Tuesday"]["1"], participation_per_day["Wednesday"]["1"], participation_per_day["Thursday"]["1"], participation_per_day["Friday"]["1"], participation_per_day["Saturday"]["1"], participation_per_day["Sunday"]["1"]]
    rects1 = ax.bar(ind, first_tries, width, color='red')
    second_tries = [participation_per_day["Monday"]["2"], participation_per_day["Tuesday"]["2"], participation_per_day["Wednesday"]["2"], participation_per_day["Thursday"]["2"], participation_per_day["Friday"]["2"], participation_per_day["Saturday"]["2"], participation_per_day["Sunday"]["2"]]
    rects2 = ax.bar(ind+width, second_tries, width, color='blue')
    third_tries = [participation_per_day["Monday"]["3"], participation_per_day["Tuesday"]["3"], participation_per_day["Wednesday"]["3"], participation_per_day["Thursday"]["3"], participation_per_day["Friday"]["3"], participation_per_day["Saturday"]["3"], participation_per_day["Sunday"]["3"]]
    rects3 = ax.bar(ind+width*2, third_tries, width, color='yellow')
    fourth_tries = [participation_per_day["Monday"]["4"], participation_per_day["Tuesday"]["4"], participation_per_day["Wednesday"]["4"], participation_per_day["Thursday"]["4"], participation_per_day["Friday"]["4"], participation_per_day["Saturday"]["4"], participation_per_day["Sunday"]["4"]]
    rects4 = ax.bar(ind+width*3, fourth_tries, width, color='orange')
    fifth_tries = [participation_per_day["Monday"]["5"], participation_per_day["Tuesday"]["5"], participation_per_day["Wednesday"]["5"], participation_per_day["Thursday"]["5"], participation_per_day["Friday"]["5"], participation_per_day["Saturday"]["5"], participation_per_day["Sunday"]["5"]]
    rects5 = ax.bar(ind+width*4, fifth_tries, width, color='green')
    sixth_tries = [participation_per_day["Monday"]["6"], participation_per_day["Tuesday"]["6"], participation_per_day["Wednesday"]["6"], participation_per_day["Thursday"]["6"], participation_per_day["Friday"]["6"], participation_per_day["Saturday"]["6"], participation_per_day["Sunday"]["6"]]
    rects6 = ax.bar(ind+width*5, sixth_tries, width, color='brown')
    none_tries = [participation_per_day["Monday"]["X"], participation_per_day["Tuesday"]["X"], participation_per_day["Wednesday"]["X"], participation_per_day["Thursday"]["X"], participation_per_day["Friday"]["X"], participation_per_day["Saturday"]["X"], participation_per_day["Sunday"]["X"]]
    rects7 = ax.bar(ind+width*6, none_tries, width, color='black')

    ax.set_ylabel('Success Count per day')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]), ("1st Try", "2nd Try", "3rd Try", "4th Try", "5th Try", "6th Try", "7+ Try") )

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1*h, '%d'%int(h),
                    ha='center', va='bottom', fontsize = 8)


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    autolabel(rects7)

    plt.show()

def main():
    # Test functions
    # participationOverTime()
    participationHardModeOverTime()
    # participationPerDay()
    # successRatesPerDay()
    pass

if __name__ == "__main__":
    main()


