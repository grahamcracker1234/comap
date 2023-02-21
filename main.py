import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, deque
import string
from sklearn import linear_model, kernel_ridge
import math
import scipy.stats as st
import statsmodels.stats.api as sms
from itertools import islice
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def sliding_window(iterable, n):
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)

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

def confidence_interval_of_mean(column, interval=0.95, n=100, display=False) -> tuple[int, int]:
    means = np.array([column.sample(n=n).mean() for _ in range(10000)])
    low, high = st.t.interval(interval, len(means)-1, loc=np.mean(column.values), scale=st.sem(column.values))
    
    if display:
        fig, ax = plt.subplots()
        # plt.hist(means, bins=25)
        plt.hist(means, density=True, bins=25)
        locs, _ = plt.yticks()
        plt.yticks(locs,np.round(locs/len(means),3))
        plt.title("Random Sampling of Mean Rates of Hard Mode")
        plt.ylabel("Probability Density")
        plt.xlabel("Rate of Hard Mode")
        plt.axvline(x=low, color="red", label="lower bound")
        plt.axvline(x=high, color="red", label="upper bound")
    return (low, high)

def test_confidence_interval(interval, test, display=False) -> bool:
    low, high = interval
    is_not_significant = low < test < high
    if display:
        if is_not_significant:
            print(f"{test} is within interval ({low}, {high})")
        else:
            print(f"{test} is not within interval ({low}, {high})")
    return not is_not_significant

def hard_rate_attributes(df, verbose=False) -> None:
    double_letters = {word for word in df.word if len(set(list(word))) < 5}
    double_letters_df = df[df.word.isin(double_letters)]
    single_letters_df = df[~df.word.isin(double_letters)]
    interval = confidence_interval_of_mean(df.hard_rate, display=verbose)
    
    significance = test_confidence_interval(interval, df.hard_rate.mean(), display=verbose)
    print(f"population hard_rate mean: {'significant' if significance else 'insignificant'}")
    
    significance = test_confidence_interval(interval, double_letters_df.hard_rate.mean(), display=verbose)
    print(f"double_letters hard_rate mean: {'significant' if significance else 'insignificant'}")
    
    significance = test_confidence_interval(interval, single_letters_df.hard_rate.mean(), display=verbose)
    print(f"single_letters hard_rate mean: {'significant' if significance else 'insignificant'}")
    
    df["vowel_count"] = df.word.apply(lambda word: sum(char.lower() in "aeiou" for char in word))
    for i, vowel_count in enumerate(df[df.vowel_count == i] for i in range(6)):
        if np.isnan(vowel_count.hard_rate.mean()):
            print(f"{i}-vowel hard_rate mean: inconclusive")
        else:
            significance = test_confidence_interval(interval, vowel_count.hard_rate.mean(), display=verbose)
            print(f"{i}-vowel hard_rate mean: {'significant' if significance else 'insignificant'}")
    
    rare_letters = "jqxz"
    rare_letters_df = df[df.word.apply(lambda word: any(char.lower() in rare_letters for char in word))]
    significance = test_confidence_interval(interval, rare_letters_df.hard_rate.mean(), display=verbose)
    print(f"rare_letters hard_rate mean: {'significant' if significance else 'insignificant'}")
    
    significance = test_confidence_interval(interval, df.nlargest(50, "mean_tries").hard_rate.mean(), display=verbose)
    print(f"hardest_words hard_rate mean: {'significant' if significance else 'insignificant'}")
    
# def difficulty_analysis(df, verbose=False) -> None:
#     # mean_tries_interval = confidence_interval_of_mean(df.mean_tries)
#     # success_rate_interval = confidence_interval_of_mean(df.success_rate)
    
#     # freq = Counter(["".join(window) for word in df.word for window in sliding_window(word, 2)])
#     # freq_df = pd.DataFrame(freq.values(), index=freq.keys(), columns=["count"])
#     # freq_df.sort_values("count", ascending=False, inplace=True)
#     # common_pairs = freq_df.head(10).index.values
#     # print("common_pairs:", common_pairs)
    
#     # common_pairs_df = df[df.word.apply(lambda word: any(pair in word for pair in common_pairs))]
#     # significance = test_confidence_interval(mean_tries_interval, common_pairs_df.mean_tries.mean(), display=verbose)
#     # print(f"common_pairs mean_tries mean: {'significant' if significance else 'insignificant'}")
#     # significance = test_confidence_interval(success_rate_interval, common_pairs_df.success_rate.mean(), display=verbose)
#     # print(f"common_pairs success_rate mean: {'significant' if significance else 'insignificant'}")
#     # print()
    
#     rare_letters = "jqxz"
#     rare_letters_df = df[df.word.apply(lambda word: any(char.lower() in rare_letters for char in word))]
#     print("rare_letters:")
#     significance_analysis(df, rare_letters_df)
#     # significance = test_confidence_interval(mean_tries_interval, rare_letters_df.mean_tries.mean(), display=verbose)
#     # print(f"rare_letters mean_tries mean: {'significant' if significance else 'insignificant'}")   
#     # significance = test_confidence_interval(success_rate_interval, rare_letters_df.success_rate.mean(), display=verbose)
#     # print(f"rare_letters success_rate mean: {'significant' if significance else 'insignificant'}")
#     # print()

#     # df["vowel_count"] = df.word.apply(lambda word: sum(char.lower() in "aeiou" for char in word))
#     # for i, vowel_count in enumerate(df[df.vowel_count == i] for i in range(6)):
#     #     if np.isnan(vowel_count.mean_tries.mean()):
#     #         print(f"{i}-vowel mean_tries mean: inconclusive")
#     #     else:
#     #         significance = test_confidence_interval(mean_tries_interval, vowel_count.mean_tries.mean(), display=verbose)
#     #         print(f"{i}-vowel mean_tries mean: {'significant' if significance else 'insignificant'}")
#     #     if np.isnan(vowel_count.success_rate.mean()):
#     #         print(f"{i}-vowel success_rate mean: inconclusive")
#     #     else:
#     #         significance = test_confidence_interval(success_rate_interval, vowel_count.success_rate.mean(), display=verbose)
#     #         print(f"{i}-vowel success_rate mean: {'significant' if significance else 'insignificant'}")
#     #     print()

def letter_analysis(df, verbose=False) -> None:
    freq = np.array([0] * 26)

    for word in df.word:
        for char in word:
            freq[index_of_char(char)] += 1
    
    for i, letter in enumerate(string.ascii_uppercase):
        letter_df = df[df.word.apply(lambda word: letter in word)]
        print(f"letter-{letter}:")
        significance_analysis(df, letter_df)

def difficulty_analysis(df, verbose=False) -> None:    
    common_pairs_freq = Counter(["".join(window) for word in df.word for window in sliding_window(word, 2)])
    common_pairs_freq_df = pd.DataFrame(common_pairs_freq.values(), index=common_pairs_freq.keys(), columns=["count"])
    common_pairs_freq_df.sort_values("count", ascending=False, inplace=True)
    common_pairs = common_pairs_freq_df.head(10).index.values
    common_pairs_df = df[df.word.apply(lambda word: any(pair in word for pair in common_pairs))]
    print("common_pairs:", common_pairs)
    significance_analysis(df, common_pairs_df)
    
    rare_letters = "jqxz"
    rare_letters_df = df[df.word.apply(lambda word: any(char.lower() in rare_letters for char in word))]
    print("rare_letters:", list(rare_letters))
    significance_analysis(df, rare_letters_df)

    df["vowel_count"] = df.word.apply(lambda word: sum(char.lower() in "aeiou" for char in word))
    for i, vowel_count_df in enumerate(df[df.vowel_count == i] for i in range(4)):
        print(f"{i}-vowel:")
        significance_analysis(df, vowel_count_df)
    
    double_letters = {word for word in df.word if len(set(list(word))) < 5}
    double_letters_df = df[df.word.isin(double_letters)]
    print("double_letters:")
    significance_analysis(df, double_letters_df)
    
    complex_double_letters = {word for word in df.word if len(set(list(word))) < 4}
    complex_double_letters_df = df[df.word.isin(complex_double_letters)]
    print("complex_double_letters:")
    significance_analysis(df, complex_double_letters_df)
    
def significance_analysis(population, sample, columns=None, verbose=False):
    columns = columns or ["try_1", "try_2", "try_3", "try_4", "try_5", "try_6", "try_fail", "success_rate", "mean_tries"]
    for column in columns:
        interval = confidence_interval_of_mean(population[column])
        significance = test_confidence_interval(interval, sample[column].mean(), display=verbose)
        print(f"sample {column} mean: {'significant' if significance else 'insignificant'}")

# def success_rate_predictor(df):
#     mask = np.random.rand(len(df)) < 0.8
#     training = df[mask]
#     testing = df[~mask]
    
#     X = np.array(list(zip(training.repeat_letters.values, training.rare_letters.values, training.vowels.values)))
#     y = training.success_rate.values

#     model = linear_model.LinearRegression()
#     model.fit(X, y)
    
#     X = np.array(list(zip(testing.repeat_letters.values, testing.rare_letters.values, testing.vowels.values)))
#     y = model.predict(X)
    
#     fig, ax = plt.subplots()
#     # ax.scatter(df.contest_number, df.success_rate, marker="o", color="blue")
#     ax.scatter(testing.contest_number, testing.success_rate, marker="o", color="green")
#     ax.scatter(testing.contest_number, y, marker=".", color="red")
    
#     mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, testing.success_rate)) / len(y)
#     print(mean_absolute_error)
    
#     words = ["avail", "parer", "judge"]
#     X = np.array([[repeat_letters(word), rare_letters(word), vowels(word)] for word in words])
#     # print(model.predict(np.array([[1, 0, 3]]))) # avail
#     # print(model.predict(np.array([[1, 0, 3]]))) # parer
#     # print(model.predict(np.array([[0, 1, 2]]))) # judge
#     print(model.predict(X))
#     print(df.word.apply(lambda word: True))

#     # plt.show()
   
def mean_tries_predictor(df):
    np.random.seed(0)
    mask = np.random.rand(len(df)) < 0.8
    training = df
    testing = df[~mask]
    
    X = np.array(list(zip(
        training.repeat_letters.values,
        training.rare_letters.values,
        training.vowels.values,
        training.freq_score.values,
        training.word_freq.values,
    )))
    y = training.mean_tries.values

    poly = PolynomialFeatures(3, interaction_only=True)
    poly.fit(X)
    features = poly.get_feature_names_out(input_features=["repeat_letters","rare_letter","vowels","freq_score","word_freq"])
    # print(len(features), features)
    model = make_pipeline(PolynomialFeatures(3, interaction_only=True), linear_model.LinearRegression())
    # model = linear_model.LinearRegression()
    means_mean = training.mean_tries.mean()
    model.fit(X, y)
    # model.fit(X, y, linearregression__sample_weight=((training.mean_tries - means_mean) ** 2).values)
    # model.fit(X, y, sample_weight=(((training.mean_tries - means_mean) ** 2) ** 1).values)
    print("score", model.score(X, y))
    
    X = np.array(list(zip(
        testing.repeat_letters.values,
        testing.rare_letters.values, 
        testing.vowels.values,
        training.freq_score.values,
        training.word_freq.values,
    )))
    y = model.predict(X)
    
        
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    # ax.scatter(df.contest_number, df.mean_tries, marker="o", color="blue")
    ax.vlines(testing.contest_number, y, testing.mean_tries, zorder=0, color="gray")
    ax.scatter(testing.contest_number, testing.mean_tries, marker="o", color="green", zorder=10, label="True mean")
    ax.scatter(testing.contest_number, y, marker=",", color="red", zorder=10, label="Predicted mean")
    plt.title("Mean Attempts Model vs. Actual Mean Attempts")
    plt.xlabel("Contest Number")
    plt.ylabel("Mean Number of Attempts")
    plt.legend(loc="upper right")
    # ax.errorbar(testing.contest_number, (testing.mean_tries + y) / 2, yerr=(testing.mean_tries - y) / 2)
    
    mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, testing.mean_tries)) / len(y)
    root_mean_square_error = (sum((y_true - y_i)*(y_true - y_i) for y_i, y_true in zip(y, testing.mean_tries)) / len(y)) ** 0.5
    print("mean_absolute_error", mean_absolute_error)
    print("root_mean_square_error", root_mean_square_error)
    
    words = ["eerie"]
    freq_df = pd.read_csv("./letter_freq.csv")
    word_freq_df = pd.read_csv("./unigram_freq.csv")
    X = np.array([[
        repeat_letters(word), 
        rare_letters(word), 
        vowels(word),
        freq_score(freq_df, word),
        word_freq(word_freq_df, word),
    ] for word in words])
    print(words)
    print(model.predict(X))
    # print(df[df.word.apply(lambda word: word in words)].mean_tries.values)

    # hardest_words_df = pd.concat([df.nlargest(10, "mean_tries"), df.nsmallest(10, "mean_tries")], ignore_index=True, axis=0)
    # X = np.array(list(zip(
    #     hardest_words_df.repeat_letters.values, 
    #     hardest_words_df.rare_letters.values, 
    #     hardest_words_df.vowels.values,
    #     hardest_words_df.freq_score.values,
    #     hardest_words_df.word_freq.values,
    # )))
    # y = model.predict(X)
    # mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)
    # root_mean_square_error = (sum((y_true - y_i)*(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)) ** 0.5

    # print("mean_absolute_error", mean_absolute_error)
    # print("root_mean_square_error", root_mean_square_error)
    # print(*zip(hardest_words_df.word.values, y, hardest_words_df.mean_tries.values), sep="\n")

    # plt.show()
    
def sd_tries_predictor(df):
    np.random.seed(0)
    mask = np.random.rand(len(df)) < 0.8
    training = df
    testing = df[~mask]
    
    X = np.array(list(zip(
        training.repeat_letters.values,
        training.rare_letters.values,
        training.vowels.values,
        training.freq_score.values,
        training.word_freq.values,
    )))
    y = training.sd_tries.values

    poly = PolynomialFeatures(3, interaction_only=True)
    poly.fit(X)
    features = poly.get_feature_names_out(input_features=["repeat_letters","rare_letter","vowels","freq_score","word_freq"])
    # print(len(features), features)
    model = make_pipeline(PolynomialFeatures(3, interaction_only=True), linear_model.LinearRegression())
    # model = linear_model.LinearRegression()
    means_mean = training.sd_tries.mean()
    model.fit(X, y)
    # model.fit(X, y, linearregression__sample_weight=((training.mean_tries - means_mean) ** 2).values)
    # model.fit(X, y, sample_weight=(((training.mean_tries - means_mean) ** 2) ** 1).values)
    print("score", model.score(X, y))
    
    X = np.array(list(zip(
        testing.repeat_letters.values,
        testing.rare_letters.values, 
        testing.vowels.values,
        training.freq_score.values,
        training.word_freq.values,
    )))
    y = model.predict(X)
    
        
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    # ax.scatter(df.contest_number, df.mean_tries, marker="o", color="blue")
    ax.vlines(testing.contest_number, y, testing.mean_tries, zorder=0, color="gray")
    ax.scatter(testing.contest_number, testing.mean_tries, marker="o", color="green", zorder=10, label="True mean")
    ax.scatter(testing.contest_number, y, marker=",", color="red", zorder=10, label="Predicted mean")
    plt.title("Mean Attempts Model vs. Actual Mean Attempts")
    plt.xlabel("Contest Number")
    plt.ylabel("Mean Number of Attempts")
    plt.legend(loc="upper right")
    # ax.errorbar(testing.contest_number, (testing.mean_tries + y) / 2, yerr=(testing.mean_tries - y) / 2)
    
    mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, testing.sd_tries)) / len(y)
    root_mean_square_error = (sum((y_true - y_i)*(y_true - y_i) for y_i, y_true in zip(y, testing.sd_tries)) / len(y)) ** 0.5
    print("mean_absolute_error", mean_absolute_error)
    print("root_mean_square_error", root_mean_square_error)
    
    words = ["eerie"]
    freq_df = pd.read_csv("./letter_freq.csv")
    word_freq_df = pd.read_csv("./unigram_freq.csv")
    X = np.array([[
        repeat_letters(word), 
        rare_letters(word), 
        vowels(word),
        freq_score(freq_df, word),
        word_freq(word_freq_df, word),
    ] for word in words])
    print(words)
    print(model.predict(X))
    # print(df[df.word.apply(lambda word: word in words)].sd_tries.values)

    # hardest_words_df = pd.concat([df.nlargest(10, "mean_tries"), df.nsmallest(10, "mean_tries")], ignore_index=True, axis=0)
    # X = np.array(list(zip(
    #     hardest_words_df.repeat_letters.values, 
    #     hardest_words_df.rare_letters.values, 
    #     hardest_words_df.vowels.values,
    #     hardest_words_df.freq_score.values,
    #     hardest_words_df.word_freq.values,
    # )))
    # y = model.predict(X)
    # mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)
    # root_mean_square_error = (sum((y_true - y_i)*(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)) ** 0.5

    # print("mean_absolute_error", mean_absolute_error)
    # print("root_mean_square_error", root_mean_square_error)
    # print(*zip(hardest_words_df.word.values, y, hardest_words_df.mean_tries.values), sep="\n")

    # plt.show()
  
def success_rate_predictor(df):
    np.random.seed(0)
    mask = np.random.rand(len(df)) < 0.8
    training = df
    testing = df[~mask]
    
    X = np.array(list(zip(
        training.repeat_letters.values,
        training.rare_letters.values,
        training.vowels.values,
        training.freq_score.values,
        training.word_freq.values,
    )))
    y = training.success_rate.values

    poly = PolynomialFeatures(3, interaction_only=True)
    poly.fit(X)
    features = poly.get_feature_names_out(input_features=["repeat_letters","rare_letter","vowels","freq_score","word_freq"])
    # print(len(features), features)
    # model = make_pipeline(PolynomialFeatures(3, interaction_only=True), linear_model.LinearRegression())
    model = linear_model.LinearRegression()
    means_mean = training.success_rate.mean()
    model.fit(X, y)
    # model.fit(X, y, linearregression__sample_weight=((training.mean_tries - means_mean) ** 2).values)
    model.fit(X, y, sample_weight=(((training.mean_tries - means_mean) ** 2) ** 1).values)
    print("score", model.score(X, y))
    
    X = np.array(list(zip(
        testing.repeat_letters.values,
        testing.rare_letters.values, 
        testing.vowels.values,
        training.freq_score.values,
        training.word_freq.values,
    )))
    y = model.predict(X)
    
        
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    # ax.scatter(df.contest_number, df.mean_tries, marker="o", color="blue")
    ax.vlines(testing.contest_number, y, testing.success_rate, zorder=0, color="gray")
    ax.scatter(testing.contest_number, testing.success_rate, marker="o", color="green", zorder=10, label="True mean")
    ax.scatter(testing.contest_number, y, marker=",", color="red", zorder=10, label="Predicted mean")
    plt.title("Mean Attempts Model vs. Actual Mean Attempts")
    plt.xlabel("Contest Number")
    plt.ylabel("Mean Number of Attempts")
    plt.legend(loc="upper right")
    # ax.errorbar(testing.contest_number, (testing.mean_tries + y) / 2, yerr=(testing.mean_tries - y) / 2)
    
    mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, testing.success_rate)) / len(y)
    root_mean_square_error = (sum((y_true - y_i)*(y_true - y_i) for y_i, y_true in zip(y, testing.success_rate)) / len(y)) ** 0.5
    print("mean_absolute_error", mean_absolute_error)
    print("root_mean_square_error", root_mean_square_error)
    
    words = ["eerie"]
    freq_df = pd.read_csv("./letter_freq.csv")
    word_freq_df = pd.read_csv("./unigram_freq.csv")
    X = np.array([[
        repeat_letters(word), 
        rare_letters(word), 
        vowels(word),
        freq_score(freq_df, word),
        word_freq(word_freq_df, word),
    ] for word in words])
    print(words)
    print(model.predict(X))
    # print(df[df.word.apply(lambda word: word in words)].sd_tries.values)

    # hardest_words_df = pd.concat([df.nlargest(10, "mean_tries"), df.nsmallest(10, "mean_tries")], ignore_index=True, axis=0)
    # X = np.array(list(zip(
    #     hardest_words_df.repeat_letters.values, 
    #     hardest_words_df.rare_letters.values, 
    #     hardest_words_df.vowels.values,
    #     hardest_words_df.freq_score.values,
    #     hardest_words_df.word_freq.values,
    # )))
    # y = model.predict(X)
    # mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)
    # root_mean_square_error = (sum((y_true - y_i)*(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)) ** 0.5

    # print("mean_absolute_error", mean_absolute_error)
    # print("root_mean_square_error", root_mean_square_error)
    # print(*zip(hardest_words_df.word.values, y, hardest_words_df.mean_tries.values), sep="\n")

    # plt.show()
    
# def mean_tries_predictor(df):
#     np.random.seed(0)
#     mask = np.random.rand(len(df)) < 0.8
#     training = df
#     testing = df[~mask]
    
#     X = np.array(list(zip(
#         training.repeat_letters.values,
#         training.rare_letters.values,
#         training.vowels.values,
#         training.freq_score.values,
#         training.word_freq.values,
#     )))
#     y = training.mean_tries.values

#     model = linear_model.LinearRegression()
#     means_mean = training.mean_tries.mean()
#     # sds_mean = training.sd_tries.mean()
#     # model.fit(X, y, sample_weight=training.mean_tries.apply(lambda t: 2 ** t).values)
#     model.fit(X, y, sample_weight=(((training.mean_tries - means_mean) ** 2 / training.sd_tries ** 2) ** 1).values)
#     print(model.score(X, y))
    
#     X = np.array(list(zip(
#         testing.repeat_letters.values, 
#         testing.rare_letters.values, 
#         testing.vowels.values,
#         training.freq_score.values,
#         training.word_freq.values,
#     )))
#     y = model.predict(X)
    
#     fig, ax = plt.subplots()
#     # ax.scatter(df.contest_number, df.mean_tries, marker="o", color="blue")
#     ax.vlines(testing.contest_number, y, testing.mean_tries)
#     ax.scatter(testing.contest_number, testing.mean_tries, marker="o", color="green")
#     ax.scatter(testing.contest_number, y, marker=",", color="red")
#     # ax.errorbar(testing.contest_number, (testing.mean_tries + y) / 2, yerr=(testing.mean_tries - y) / 2)
    
#     mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, testing.mean_tries)) / len(y)
#     print(mean_absolute_error)
    
#     words = ["eerie", "nymph"]
#     freq_df = pd.read_csv("./letter_freq.csv")
#     word_freq_df = pd.read_csv("./unigram_freq.csv")
#     X = np.array([[
#         repeat_letters(word), 
#         rare_letters(word), 
#         vowels(word),
#         freq_score(freq_df, word),
#         word_freq(word_freq_df, word),
#     ] for word in words])
#     print(words)
#     print(model.predict(X))
#     # print(df[df.word.apply(lambda word: word in words)].mean_tries.values)

#     hardest_words_df = df.nlargest(10, "mean_tries")
#     X = np.array(list(zip(
#         hardest_words_df.repeat_letters.values, 
#         hardest_words_df.rare_letters.values, 
#         hardest_words_df.vowels.values,
#         hardest_words_df.freq_score.values,
#         hardest_words_df.word_freq.values,
#     )))
#     y = model.predict(X)
#     mean_absolute_error = sum(abs(y_true - y_i) for y_i, y_true in zip(y, hardest_words_df.mean_tries)) / len(y)
#     print(mean_absolute_error)
#     print(*zip(df.nlargest(10, "mean_tries").word.values, y, hardest_words_df.mean_tries.values), sep="\n")

#     plt.show()

def repeat_letters(word):
    return sum(count - 1 for count in Counter(word).values())

def rare_letters(word):
    return sum(char.lower() in "jqxzy" for char in word)

def vowels(word):
    return sum(char.lower() in "aeiou" for char in word)

def freq_score(df, word):
    freq = df.freq.values
    return sum(freq[index_of_char(char)] for char in word) 

def word_freq(df, word):
    freq = df[df.word == word]["count"].values
    return freq[0] if len(freq) > 0 else 0

def generate_data():
    df = pd.read_csv("./data.csv", parse_dates=["date"])

    df["repeat_letters"] = df.word.apply(repeat_letters)

    df["rare_letters"] = df.word.apply(rare_letters)
    
    df["vowels"] = df.word.apply(vowels)
    
    freq = np.array([0] * 26)
    for word in df.word:
        for char in word:
            freq[index_of_char(char)] += 1
    
    freq_df = pd.DataFrame(freq, index=list(string.ascii_lowercase), columns=["freq"])
    freq_df.to_csv("./letter_freq.csv")
    
    df["freq_score"] = df.word.apply(lambda word: freq_score(freq_df, word))
    
    tries = [df[f"try_{i + 1}"] for i in range(6)]
    df["sd_tries"] = (sum(_try * ((i + 1) - df.mean_tries) ** 2 for i, _try in enumerate(tries)) / df.success_rate) ** 0.5
    # print(df.sd_tries)
    
    word_freq_df = pd.read_csv("./unigram_freq.csv")
    df["word_freq"] = df.word.apply(lambda word: word_freq(word_freq_df, word))
    print(df)
    
    df.to_csv("./new_data.csv")

if __name__ == "__main__":
    df = pd.read_csv("./new_data.csv", parse_dates=["date"])
    # generate_data()
    # success_rate_predictor(df)
    
    
    # mean_tries_predictor(df)
    # sd_tries_predictor(df)
    # success_rate_predictor(df)
    
    # s = np.random.normal(4.8, 0.916, 100000)
    mu = 4.8
    sd = 0.916
    rate = 95
    
    s = np.random.normal(mu, sd, 1000000)
    bins = plt.hist(s, density=True, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    print(bins[0])
    attempts = (bins[0] / bins[0].sum() * rate).round()
    print(attempts, attempts.sum())
    attempts = np.array([0, 1, 6, 26, 42, 21])
    print(attempts, attempts.sum())
    print(sum(attempt * (i + 1) for i, attempt in enumerate(attempts)) / attempts.sum())
    
    # print(df.nlargest(50, "mean_tries")[["word", "mean_tries"]])
    # locs, _ = plt.yticks()

    
    # plt.yticks(locs,np.round(locs/len(s),3))
    plt.title("Number of Attempts for a Successful Contest")
    plt.ylabel("Probability Density")
    plt.xlabel("Number of Attempts")
    
    # plt.scatter(df.mean_tries, df.hard_rate)
    # plt.show()
    

    # hard_rate_attributes(df, verbose=True)
    # plt.show()
    # difficulty_analysis(df, verbose=True)
    # letter_analysis(df)

    # scores = {"a": 1, "c": 3, "b": 3, "e": 1, "d": 2, "g": 2, "f": 4, "i": 1, "h": 4, "k": 5, "j": 8, "m": 3, "l": 1, "o": 1, "n": 1, "q": 10, "p": 3, "s": 1, "r": 1, "u": 1, "t": 1, "w": 4, "v": 4, "y": 4, "x": 8, "z": 10}
    # df["score"] = df.word.apply(lambda word: sum(scores[char] for char in word))
    # plt.scatter(df.score, df.mean_tries)
    # plt.ylim(1, 6)
    # plt.show()

    # freq = total_char_freq(df, display=False)
    # freq = positional_char_freq(df, display=False)
    # print(sms.DescrStatsW(means).tconfint_mean())
    # print(mean_confidence_interval(means))
    # df.hard_rate.plot.hist(density=1, bins=15, ax=ax, label="null") 
    # double_letters_df.hard_rate.plot.hist(density=1, bins=15, ax=ax, label="double") 
    # plt.show()
    
    # df["vowels_count"] = df.word.apply(lambda word: sum(char.lower() in "aeiou" for char in word))
    
    # zero_vowel_df = df[df.vowels_count == 0]
    # one_vowel_df = df[df.vowels_count == 1]
    # two_vowel_df = df[df.vowels_count == 2]
    # three_vowel_df = df[df.vowels_count == 3]
    # four_vowel_df = df[df.vowels_count == 4]
    # five_vowel_df = df[df.vowels_count == 5]
    # print(len(one_vowel_df), len(two_vowel_df))
    # # print(df.sample(n=len(zero_vowel_df)))
    # plt.show()
    # print(df.hard_rate.sample(n=len(zero_vowel_df)).corrwith(zero_vowel_df))
    
    # print("zero", zero_vowel_df.hard_rate.mean(), zero_vowel_df.results_count.mean())
    # print("one", one_vowel_df.hard_rate.mean(), one_vowel_df.results_count.mean())
    # print("two", two_vowel_df.hard_rate.mean(), two_vowel_df.results_count.mean())
    # print("three", three_vowel_df.hard_rate.mean(), three_vowel_df.results_count.mean())
    # print("four", four_vowel_df.hard_rate.mean(), four_vowel_df.results_count.mean())
    # print("five", five_vowel_df.hard_rate.mean(), five_vowel_df.results_count.mean())

    # fig, ax = plt.subplots()
    # ax.scatter(double_letters_df.mean_tries, double_letters_df.hard_rate, color="blue")
    # ax.scatter(single_letters_df.mean_tries, single_letters_df.hard_rate, color="red")
    # plt.show()
    # plt.xscale("logit")
    # plt.yscale("logit")
    # double_letters_df.plot.scatter(x="mean_tries")
    # print([word for word in df.word if len(set(list(word))) < 4])
    # print(sum(l < 5 for l in [len(set(list(word))) for word in df.word]))
    
    
    # df["success_rate"] = df.try_1 + df.try_2 + df.try_3 + df.try_4 + df.try_5 + df.try_6
    # df["total"] = df.success_rate + df.try_fail
    # df["hard_rate"] = df.hard_count / df.results_count
    # df["mean_tries"] = (df.try_1 * 1 + df.try_2 * 2 + df.try_3 * 3 + df.try_4 * 4 + df.try_5 * 5 + df.try_6 * 6) / df.success_rate
    # df = df[["date", "day", "contest_number", "word", "results_count", "hard_count", "hard_rate", "try_1", "try_2", "try_3", "try_4", "try_5", "try_6", "try_fail", "success_rate", "total", "mean_tries"]]

    # df.to_csv("./test.csv", index=None)
    
    # freq = Counter(list(np.array(df.total)))
    # freq_df = pd.DataFrame(freq.values(), index=freq.keys(), columns=["count"])
    # freq_df.sort_index(ascending=True, inplace=True)
    # # freq_df.plot.bar()
    # # df.total.plot.bar()
    # df["true_success_rate"] = df.success_rate / df.total
    # df["mean_tries"] = (df.try_1 * 1 + df.try_2 * 2 + df.try_3 * 3 + df.try_4 * 4 + df.try_5 * 5 + df.try_6 * 6) / df.success_rate
    # # df["mean_tries/true_success_rate"] = 
    # # df[["hard_rate", "mean_tries"]].plot()
    
    # df.plot.scatter(x="hard_rate", y="mean_tries")
    # df["1/results_count"] = 1 / df.results_count
    # df.plot.scatter(x="hard_rate", y="results_count")
    # df["hard_rate*results_count"] = df.hard_rate * df.results_count
    # df.plot.scatter(x="hard_rate*results_count", y="mean_tries")
    
    # x = df.contest_number.values[35:].reshape(-1, 1)
    # y = df.results_count.values[35:]
    # fig, ax = plt.subplots()
    # df.plot.scatter(x="contest_number", y="results_count", ax=ax)
    
    # clf = linear_model.GammaRegressor(solver="newton-cholesky")
    # # weight_distributer = lambda i: 1 / (math.log(i + 1) + 1)
    # weight_distributer = lambda i: 1 / (i + 1) ** 0.1
    # weights = np.array([weight_distributer(i) for i in range(len(y))])
    # print(weights, y)
    # clf.fit(x, y, sample_weight=weights)
    
    # new_x = np.array(list(range(0, 650)))
    # new_y = clf.predict(x)
    # # plt.xscale("log")
    # # plt.yscale("log")
    # ax.plot(x, new_y, color="green")
    # plt.show()
    # # # print(df.total)
