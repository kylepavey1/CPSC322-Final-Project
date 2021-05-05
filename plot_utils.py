import matplotlib.pyplot as plt

def bar_chart(labels, values, xlabel):
    plt.figure()
    plt.title("Total Count by " + xlabel)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.bar(labels, values)
    plt.xticks(labels, rotation=90, horizontalalignment="right")
    plt.show()

def pie_chart(label, values, number_of, by_value):
    plt.figure()
    plt.title("Total " + number_of + " by " + by_value)
    plt.pie(values, labels=label, autopct="%1.1f%%")
    plt.show()

def histogram_chart(data, data2, name, name2, bin_size):
    plt.figure()
    if data2 is None or name2 is None:
        plt.title("Distribution of " + name)
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.hist(data, bins=bin_size)
    else:
        plt.title("Distribution of " + name + " vs " + "distribution of " + name2)
        plt.xlabel(name + " in orange, " + name2 + " in blue")
        plt.ylabel("Count")
        plt.hist(data, bins=bin_size, color="orange")
        plt.hist(data2, bins=bin_size, color="blue")
        plt.savefig("line_chart.pdf")
    plt.show()

def linear_regression(x_name, y_name, x, y, m, b):
    plt.figure()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(x_name + " vs " + y_name + " correlation")
    plt.grid(True)
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
    plt.show()

def box_plot(distributions, labels, name):
    plt.figure()
    plt.boxplot(distributions)
    plt.title(name + " Ratings by Genre")
    plt.xticks(list(range(1, len(labels) + 1)), labels, rotation=90, horizontalalignment="right")
    plt.annotate("$\mu=100$", xy=(0.5, 0.8), xycoords="data", horizontalalignment="center")
    plt.annotate("$\mu=100$", xy=(0.5, 0.9), xycoords="axes fraction", horizontalalignment="center", color="blue")
    plt.show()