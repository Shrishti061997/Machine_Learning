import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data file 
file_path = 'pima-indians-diabetes.xlsx'
df = pd.read_excel(file_path)
total_size = len(df)
Not_diabetes_df = df[df['Outcome'] == 0]
Diabetes_df = df[df['Outcome'] == 1 ]

prediction_accuracy={}

for n in [40,80,120,160,200]:
    total_test_cases = 0
    correct_predictions = 0
    for i in range(1000):

        #Prepare training dataset
        Selected_diabetes = Diabetes_df.sample(n=n)
        Selected_not_diabetes = Not_diabetes_df.sample(n=n)
        train_df = pd.concat([Selected_diabetes,Selected_not_diabetes])
        X = train_df[train_df.columns[:-1]]
        t = train_df[train_df.columns[-1]]

        #Solve linear regression
        X_T = X.transpose()
        B = (X_T.dot(t)).to_numpy()
        A = (X_T.dot(X)).to_numpy()
        w = np.linalg.solve(A,B)

        #Testing accuracy
        test_df = (df.drop(train_df.index))
        test_x = (test_df[test_df.columns[:-1]]).to_numpy()
        Actual_outcome = (test_df[test_df.columns[-1]]).to_numpy()
        output_val = np.dot(test_x , w)
        predicted_outcome = [0 if data < 0.5 else 1 for data in output_val]
    
        for index in range (len (predicted_outcome)):
            if predicted_outcome[index] == Actual_outcome[index]:
                correct_predictions += 1
        total_test_cases += (total_size-(2 * n))
       
    prediction_accuracy[n] = (correct_predictions / total_test_cases) * 100


print (prediction_accuracy)

#plot graph
plt.plot(prediction_accuracy.keys(), prediction_accuracy.values(), marker='o')
plt.title("Accuracy vs n")
plt.xlabel("n")
plt.ylabel("Accuracy Rate")
plt.savefig("Prima_indian_diabetes_plot.jpg", format='jpeg', dpi=300)

    

