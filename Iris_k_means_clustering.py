import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Load data file 
file_path = 'Iris.xlsx'
df = pd.read_excel(file_path)
df = (df[df.columns[:-1]])
total_size = len(df)

epsilon = 0.00001
Cluster_centres = {}
Class_count = 3
Selected_clusteroids = df.sample(n=Class_count)

Classification = {}
Class_vector = {}

for i in range (Class_count):
    Cluster_centres[i] = Selected_clusteroids.iloc[i].to_numpy()
    Classification[i]=[]



dataset = df.to_numpy()
cost_trace = []

while True:
    #Clustering of the data
    for i in range  (total_size):
        sample_data = dataset[i]

        minimum_class = 0
        min_distance = np.inf

        for j in range (Class_count):
            distance =  np.sum(np.abs(sample_data - Cluster_centres[j]))
            if distance < min_distance:
                min_distance = distance
                minimum_class = j

        Classification[minimum_class].append(sample_data)
        Class_vector[i] = minimum_class

    #Updating the clusteroids
    for i in range (Class_count):
        Cluster_centres[i] = np.mean(Classification[i], axis=0)

    total_diff = 0
    
    #Computing cost funtion 
    for i in range (total_size):
        sample_data = dataset[i]

        Selected_clusteroid = Cluster_centres[Class_vector[i]]
        diff = np.sum(np.abs(sample_data - Selected_clusteroid)) ** 2
        total_diff = total_diff + diff

    cost = total_diff/total_size
    cost_trace.append(cost)

    if len(cost_trace) > 1 :
        print(f"diff {cost_trace[-2] - cost_trace[-1]}")
        if cost_trace[-2] - cost_trace[-1] < epsilon:
            break

print(cost_trace)
     
#plot graph
plt.plot(cost_trace)
plt.title("J vs Iter")
plt.xlabel("Iter_count")
plt.ylabel("J")
plt.savefig("Iris.jpg", format='jpeg', dpi=300)




