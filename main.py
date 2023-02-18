import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import string

def index_of_char(char: str) -> int:
    if char.isupper():
        return ord(char) - ord("A")
    return ord(char) - ord("a")

def total_char_freq(df, display=False):
    freq = np.array([0] * 26)

    for word in df.word:
        for char in word:
            freq[index_of_char(char)] += 1

    if display:
        freq_df = pd.DataFrame(freq, index=list(string.ascii_lowercase), columns=["count"])
        freq_df.sort_index(ascending=True, inplace=True)
        freq_df.plot.bar()
    
    return freq
    
def positional_char_freq(df, display=False):
    freq = np.array([[0] * 5] * 26)
    for word in df.word:
        for j, char in enumerate(word):
            freq[index_of_char(char)][j] += 1
            
    freq_df = pd.DataFrame(freq, index=list(string.ascii_lowercase), columns=["1st", "2nd", "3rd", "4th", "5th"])
    
    if display:
        freq_df.plot.bar()
        
    return freq_df
    
if __name__ == "__main__":
    df = pd.read_csv("./data.csv", parse_dates=["date"])
    # freq = total_char_freq(df, display=False)
    # print(freq)
    # freq = positional_char_freq(df, display=False)
    # print(freq)
    # print(len(df.word))
    # print([l < 5 for word in df.word if len(set(list(word))) < 5])
    # print(sum(l < 5 for l in [len(set(list(word))) for word in df.word]))
    # plt.show()
    df["success_rate"] = df.try_1 + df.try_2 + df.try_3 + df.try_4 + df.try_5 + df.try_6
    df["total"] = df.success_rate + df.try_fail
    freq = Counter(list(np.array(df.total)))
    freq_df = pd.DataFrame(freq.values(), index=freq.keys(), columns=["count"])
    freq_df.sort_index(ascending=True, inplace=True)
    # freq_df.plot.bar()
    # df.total.plot.bar()
    df["true_success_rate"] = df.success_rate / df.total
    df["mean_tries"] = (df.try_1 * 1 + df.try_2 * 2 + df.try_3 * 3 + df.try_4 * 4 + df.try_5 * 5 + df.try_6 * 6) / df.success_rate
    # df["mean_tries/true_success_rate"] = 
    df["hard_rate"] = df.hard_count / df.results_count
    df[["hard_rate", "mean_tries"]].plot()
    # df.plot.scatter(x="hard_rate", y="mean_tries")
    
    plt.show()
    # print(df.total)
