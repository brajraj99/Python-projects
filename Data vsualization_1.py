import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(xdata, ydata, zdata):
    
    """ For plotting the AQI of Beijing and Shanghai during different months """
    
    plt.plot(xdata,ydata,label = 'Shanghai 2015-2016', marker = 'o')
    plt.plot(xdata,zdata,label = 'Shanghai 2014-2015', marker = 'o', linestyle='dashed')
    plt.xlabel('Month', fontsize = 12)
    plt.ylabel('AQI', fontsize = 12)
    plt.title('AQI of Beijing during different months of 2014 and 2015', fontsize = 15)

def boxplot(isub, xdata):
    
    """ For Barplot showing AQI of different Cities in different months of 2015 """
    
    plt.subplot(2, 2, isub)
    plt.boxplot(xdata, labels = [label[isub-1]])
    plt.ylabel('AQI', fontsize = 12)

def barplot(datax, br):
    
    """ For Boxplot of AQI of  Beijing, Shanghai, Guangzhou, and Hangzhou in 2015 """
    
    plt.bar(brn[br-1], datax, width = barwidth, label = label[br-1], edgecolor = 'grey')    

def histogram(histdata, i):
    
    """ For plotting Histogram """
    
    plt.hist(histdata, label = histleg[i-1], bins = 20, alpha = alpha[i-1], edgecolor = 'black')
    plt.xlabel('Air Quality Index', fontsize = 12)
    plt.ylabel('Number of occurances', fontsize = 12)
    

# Importing and reading the csv file
data = pd.read_csv('china_aqi_2015_compare.csv')

# Extracting the required data out of the dataset
data_grp = data.groupby('Station')
bei = data_grp.get_group('Beijing')
sha = data_grp.get_group('Shanghai')
han = data_grp.get_group('Hangzhou')
che = data_grp.get_group('Chengdu')
gua = data_grp.get_group('Guangzhou')

xx = sha['Month']
yy = sha['RawValue']
zz = sha['LastYearRawValue']

aa = bei['Month']
bb = bei['RawValue']
cc = sha['RawValue']
dd = gua['RawValue']
ee = han['RawValue']

label = ['Beijing', 'Shanghai', 'Guangzhou', 'Hangzhou']
colour = ['blue', 'orange', 'red', 'green']
histleg = ['China 2014-2015', 'China 2015-2016']
alpha = [1, 0.6]

# Line Plot of AQI of Beijing and Shanghai during different months
plt.figure(figsize = (12, 8), dpi = 144)
plot(xx,yy,zz)
plt.legend(fontsize = 12)
plt.savefig("Line plot.jpg")
plt.show()

# Histogram of AQI of China in 2014 and 2015 for all the cities combined
plt.figure(figsize = (12, 8), dpi = 144)
histogram(data['LastYearRawValue'], 1)
histogram(data['RawValue'], 2)
plt.legend(fontsize = 12)
plt.title('Histogram of AQI of China in 2014 and 2015 for all the cities combined',
          fontsize = 15)
plt.savefig("Histogram.jpg")
plt.show()

# Barplot showing AQI of different Cities in different months of 2015
# Setting the bar width and spacing of bars on the plot
barwidth = 0.2
br1 = np.arange(len(aa))
br2 = [x + barwidth for x in br1]
br3 = [x + barwidth for x in br2]
brn = [br1, br2, br3]

plt.figure(figsize=(12, 8), dpi = 144)
barplot(bb, 1)
barplot(cc, 2)
barplot(dd, 3)
plt.xticks([r + barwidth for r in range(len(aa))], aa) # setting tick location
plt.xlabel('Month', fontsize = 12)
plt.ylabel('AQI', fontsize = 12)
plt.title('AQI of Beijing, Shanghai & Guangzhou from Feb 2015 to Jan 2016', fontsize = 15)
plt.legend(fontsize = 12)
plt.savefig('Barplot.jpg')
plt.show()

# Boxplot of AQI of Beijing, Shanghai, Guangzhou, and Hangzhou
plt.figure(figsize = (12, 8), dpi = 144)
plt.suptitle('Boxplot of AQI of Beijing, Shanghai, Guangzhou, and Hangzhou for 2015 to 2016',
             fontsize = 15)
boxplot(1, bb)
boxplot(2, cc)
boxplot(3, dd)
boxplot(4, ee)
plt.savefig('Boxplot.jpg')
plt.show()
