# file_content = ""
# with open("./data.csv", "r") as file:
#     columns, *lines = [line.strip() for line in file]
#     data = "\n".join(reversed(lines))
#     file_content = f"{columns}\n{data}"
    
# with open("./data.csv", "w") as file:
#     file.write(file_content)
    
# import pandas as pd
# from collections import Counter

# df = pd.read_csv("./data.csv", parse_dates=["date"])
# df["success_rate"] = 

# df.date = pd.to_datetime(df.date)
# # print(df.head())
# df["day"] = df.date.dt.day_name()
# df = df[["date", "day", "contest_number", "word", "results_count", "hard_count", "try_1", "try_2", "try_3", "try_4", "try_5", "try_6", "try_fail"]]
# df.to_csv("./new_data.csv", index=None)

# # chars = [''] * (len(df.index) * 5)
# # print(len(chars))
# # i = 0
# # lens = []
# # for word in df.word:
# #     lens.append((word, len(word)))
# #     for char in word:
# #         chars[i] = char
# #         i += 1
# # # print(chars)
# # # print(*enumerate(lens))
# # freq = Counter(chars)
# # print(freq)
# # print(sum(freq.values()))
