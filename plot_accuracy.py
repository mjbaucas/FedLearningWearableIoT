from matplotlib import pyplot as plt

dataX = [1,2,3,4,5,6,7,8,9,10]
dataLY = [90.50, 90.82, 90.59, 90.76, 90.94, 90.96, 91.00, 90.89, 90.73, 90.57]
dataGY = [90.50, 90.87, 91.11, 91.14, 91.75, 91.75, 91.75, 91.72, 91.72, 91.69]


def plot_data(dataX, dataLY, dataGY):
    plt.figure(0)
    plt.plot(dataX,dataGY,marker='o',label="Global")
    plt.plot(dataX,dataLY,marker='o',label="Local (Averaged)")
    plt.xlabel("Number of Clients")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.savefig('accmodel.png')

plot_data(dataX, dataLY, dataGY)