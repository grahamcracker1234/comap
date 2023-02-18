import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import string

def total_char_freq(df, display=False):
    chars: list[str] = [''] * (len(df.index) * 5)

    i = 0
    for word in df.word:
        for char in word:
            chars[i] = char
            i += 1

    freq = Counter(chars)

    if display:
        freq_df = pd.DataFrame(freq.values(), columns=["count"], index=freq.keys())
        freq_df.sort_index(ascending=True, inplace=True)
        freq_df.plot.bar()
        plt.show()
    
    return freq
    
def positional_char_freq(df, display=False):
    freq = np.array([[0] * 5] * 26)
    for word in df.word:
        for j, char in enumerate(word):
            freq[ord(char) - ord("a")][j] += 1
            
    freq_df = pd.DataFrame(freq, index=list(string.ascii_lowercase), columns=["1st", "2nd", "3rd", "4th", "5th"])
    
    if display:
        freq_df.plot.bar()
        plt.show()
        
    return freq_df
    
if __name__ == "__main__":
    df = pd.read_csv("./data.csv", parse_dates=["date"])
    freq = total_char_freq(df, display=False)
    print(freq)
    freq = positional_char_freq(df, display=True)
    print(freq)
    