from pandas import read_csv
from matplotlib import pyplot as plt

data = read_csv("data.txt")

#plt.plot(data["train"])
plt.plot(data["val"])
#plt.plot(data["loss"])

plt.show()
