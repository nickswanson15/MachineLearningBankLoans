import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv (r"https://raw.githubusercontent.com/nickswanson15/MachineLearningBankLoans/main/loan.csv")

def scale(table):
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(table), columns=table.columns)
    return scaled

def distance(list1, list2):
    zipped = list(zip(list1, list2))
    squareDifference = []
    for x, y in zipped:
        squareDifference =  squareDifference + [(x-y)**2]
    dist = sum(squareDifference)**.5
    return dist

def knn(prev, features, k):
    votes = []
    neighbors = []
    for i in range(0, len(prev)-1):
        neighbor = prev[i]
        noutcome = neighbor[-1]
        nfeatures = neighbor[0:-1]
        dist = distance(features, nfeatures)
        neighbors = neighbors + [[dist, noutcome]]
    neighbors.sort()
    for j in range(0, k-1):
        votes = votes + [neighbors[j][1]]
    return max(set(votes), key = votes.count)

def predict(prev, new, k):
    success = 0
    for i in range(0, len(new)):
        applicant = new[i]
        outcome = applicant[-1]
        features = applicant[0:-1]
        result = knn(prev, features, k)
        if outcome == result:
            success += 1
            prediction = "-correct"
        else:
            prediction = "-incorrect"
        print(f"k: {k}, Actual: {outcome}, Prediction: {result} {prediction}" )
    print(f"{success} / {len(new)} successful predictions!")

data = scale(df)
data = np.array(data)

k = int(input("Enter K (number of neighbors): "))
n = int(input("Enter N (number of new applicants to predict): "))

prevApplicants = data[:-n]
newApplicants  = data[-n:]

predict(prevApplicants, newApplicants, k)
