import json
import matplotlib.pyplot as plt

with open("log_loss.txt", "r") as file:
    data = json.load(file)
plt.plot(data)

plt.show()