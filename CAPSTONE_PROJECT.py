#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:02:58 2023

@author: mayanesen
"""

# CAPSTONE PROJECT

# %% Importing all packages I may need
import numpy as np
from scipy import stats
from scipy.stats import bootstrap  # to do bootstrap in one line!
import matplotlib.pyplot as plt  # so we can make figures
import pandas as pd
from sklearn.model_selection import train_test_split  # train test split
from sklearn.linear_model import LinearRegression  # linear regression easily
# This will allow us to do the PCA efficiently
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # this is the logistic sigmoid function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# %% RNG with my n-number: N16654133
import random
mySeed = 16654133  # This can be set to any number you want
random.seed(mySeed)  # Initialize RNG with the specified seed

# %% Loader
data = np.genfromtxt('theData.csv', delimiter=',')  # theData file
art = np.genfromtxt('theArt.csv', delimiter=',', skip_header=1)  # theArt file

# organize data
artRatings = data[:, :91]


# art type: 1 = classical; 2 = modern; 3 = non-human
classicalRatings = artRatings[:, :35]
modernRatings = artRatings[:, 35:70]
nonhumanRatings = artRatings[:, 70:]


# %% Descriptive statistics defintion (for questions 1 through 4)

def descriptive_statistics(matrix):
    median = np.median(matrix)  # median
    sd = np.std(matrix)  # standard deviation
    size = len(matrix)  # size n
    sem = sd / np.sqrt(size)  # standard error
    return np.array([median, sd, size, sem])


# %% Question 1: Is classical art more well liked than modern art?
print("Question 1: Is classical art more well liked than modern art?")

# flatten matrices to do mann whitney u test
c = classicalRatings.flatten()
m = modernRatings.flatten()

# descriptive stats
classicalStats = descriptive_statistics(c)
modernStats = descriptive_statistics(m)

# medians
classicalMedian = classicalStats[0]
modernMedian = modernStats[0]

# compare medians
if classicalMedian > modernMedian:
    print("Median classical art is more than the median of modern art")
else:
    print("Median classical art is less than the median of modern art")


# mann-whitney u test
u1, p1 = stats.mannwhitneyu(c, m)

# print the results
if p1 < 0.01:
    print("P-value:", p1)
    print("Statistically significant")
    print("Reject the null hypothesis")
else:
    print("P-value:", p1)
    print("Not statistically significant")
    print("Fail to reject the null")

print()

# histogram
plt.hist(c, bins=7, label="Classical art Ratings", color="green", alpha=0.75)
plt.hist(m, bins=7, label="Modern art Ratings", color="royalblue", alpha=0.75)
plt.legend(loc="upper left")
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title("Classical Art vs. Modern Art")
plt.show()


# %% Question 2: Is there a difference in the preference ratings for modern art vs. non-human (animals and
# computers) generated art?
print("Question 2: Is there a difference in the preference ratings for modern art vs. non-human (animals and computers) generated art?")

# flatten matrices
modernRatings = artRatings[:, 35:70]
nonhumanRatings = artRatings[:, 70:]

m = modernRatings.flatten()
nh = nonhumanRatings.flatten()

# descriptive stats
modernStats = descriptive_statistics(m)
nonhumanStats = descriptive_statistics(nh)

# medians
modernMedian = modernStats[0]
nonhumanMedian = nonhumanStats[0]

# compare medians
if nonhumanMedian > modernMedian:
    print("Median nonhuman art is more than the median of modern art")
else:
    print("Median nonhuman art is less than the median of modern art")


# mann-whitney u test
u2, p2 = stats.mannwhitneyu(m, nh)

# print the results
if p2 < 0.01:
    print("P-value:", p2)
    print("Statistically significant")
    print("Reject the null hypothesis")
else:
    print("P-value:", p2)
    print("Not statistically significant")
    print("Fail to reject the null")

print()


# histogram
plt.hist(nh, bins=7, label="Nonhuman art Ratings", color="orange", alpha=0.75)
plt.hist(m, bins=7, label="Modern art Ratings", color="royalblue", alpha=0.75)
plt.legend(loc="upper left")
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title("Nonhuman Art vs. Modern Art")
plt.show()


# %% Question 3: Do women give higher art preference ratings than men?
print("Question 3: Do women give higher art preference ratings than men?")

# gender: 1 = man; 2 = woman; 3 = non-binary
gender = data[:, 216]

# create arrays/matrices with just the data of men and women
# prevents NaN values from being in my data too
men = []
women = []

for i in range(300):
    if gender[i] == 1:
        men.append(data[i, :92])
    elif gender[i] == 2:
        women.append(data[i, :92])

menRatings = np.array(men)
womenRatings = np.array(women)

# flatten matrices
male = menRatings.flatten()
female = womenRatings.flatten()

# descriptive stats
maleStats = descriptive_statistics(male)
femaleStats = descriptive_statistics(female)

# medians
maleMedian = maleStats[0]
femaleMedian = femaleStats[0]

# compare medians
if maleMedian > femaleMedian:
    print("Median male art preference is more than the median for women")
else:
    print("Median male art preference is less than the median for women")


# mann-whitney u test
u3, p3 = stats.mannwhitneyu(male, female)

# print the results
if p3 < 0.01:
    print("P-value:", p3)
    print("Statistically Significant")
    print("Reject the null hypothesis")
else:
    print("P-value:", p3)
    print("Not statistically significant")
    print("Fail to reject the null")


print()

# histogram
plt.hist(female, bins=7, label="Female art Ratings",
         color="deeppink", alpha=0.75)
plt.hist(male, bins=7, label="Male art Ratings", color="darkblue", alpha=0.75)
plt.legend(loc="upper left")
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title("Male vs. Female Art Preferences")
plt.show()


# %% Question 4: Is there a difference in the preference ratings of users with some art background (some art
# education) vs. none?
print("Question 4: Is there a difference in the preference ratings of users with some art background (some art education) vs. none?")

education = data[:, 219]

# collect ratings for paintings if some education (1,2,3) or no education (0)
# this takes care of rows with nan values in education column as well
someEduc = []
noEduc = []

for i in range(300):
    if education[i] == 0:
        noEduc.append(data[i, :92])
    elif education[i] in [1, 2, 3]:
        someEduc.append(data[i, :92])


# flatten matrices to do mann whitney u test
someEduc = np.array(someEduc).flatten()
noEduc = np.array(noEduc).flatten()

someStats = descriptive_statistics(someEduc)
noneStats = descriptive_statistics(noEduc)

# medians
someEducMedian = someStats[0]
noEducMedian = noneStats[0]

# compare medians
if someEducMedian > noEducMedian:
    print("Median ratings for some education is more than the median of ratings for no education")
else:
    print("Median ratings for some education is less than the median of ratings for no education")


# mann-whitney u test
u4, p4 = stats.mannwhitneyu(someEduc, noEduc)

# print the results
if p4 < 0.01:
    print("P-value:", p4)
    print("Statistically Significant")
    print("Reject the null hypothesis")
else:
    print("P-value:", p4)
    print("Not statistically significant")
    print("Fail to reject the null")


print()

# histogram
plt.hist(c, bins=7, label="Some Education Ratings", color="purple", alpha=0.7)
plt.hist(m, bins=7, label="No education Ratings",
         color="sandybrown", alpha=0.5)
plt.legend(loc="upper left")
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title("Art Preference Ratings: Some vs. No Education")
plt.show()


# %% Question 5: Build a regression model to predict art preference ratings from energy ratings only.
# Make sure to use cross-validation methods to avoid overfitting and characterize how well your model
# predicts art preference ratings.

# art preference ratings = columns
# organize data

# dependent variable: y
# columns 0 to 90
artRatings = data[:, :91]

# energy: ratings from 1 (“it calms me down a lot”) to 7 (“it agitates me a lot”)
# independent variable
# columns 91 to 181
energy = data[:, 91:182]


# use pandas to remove nan values
energyClean = pd.DataFrame(energy).dropna().to_numpy()

# to have same size data for linear regression:
# row mean: because we want mean art rating per participant
# had initially used medians but median gives discrete values, not continuous...
energyMean = np.mean(energyClean, axis=1).reshape(300, 1)  # x
artMean = np.mean(artRatings, axis=1).reshape(300, 1)  # y



# cross-validation
xTrain, xTest, yTrain, yTest = train_test_split(
    energyMean, artMean, test_size=0.5, random_state=16654133)

# build the linear regression model using training data
model1 = LinearRegression().fit(xTrain, yTrain)
b0_1, b1_1 = model1.coef_, model1.intercept_


# cross-validation: using the betas from the training dataset, but
# measuring the error with the test dataset
yHat = b0_1 * xTrain + b1_1
#yPred = b0_1 * xTest + b1_1
yPred = model1.predict(xTest)
# rmse and R^2
rmse = np.sqrt(np.mean((yPred - yTest)**2))
rSqr = model1.score(energyMean, artMean)


# scatter plot between predicted and actual score of full model:
#plt.plot(xTest, yTest, "o", markersize=4)
plt.plot(yTest, yPred, "o", markersize=4)
plt.ylabel('Predicted Art Rating')
plt.xlabel('Actual Art Rating')
plt.suptitle("Art Preference vs. Energy Ratings")
#plt.title(f'RMSE = {rmse:.3f}', fontsize=10)
plt.title(f'RMSE = {rmse:.3f}, R^2 = {rSqr:.3f}', fontsize=10)
plt.show()


# %% Question 6: Build a regression model to predict art preference ratings from energy ratings and
# demographic information. Make sure to use cross-validation methods to avoid overfitting and
# comment on how well your model predicts relative to the “energy ratings only” model.

# energy
energy = data[:, 91:182]

# gender: 1 = man; 2 = woman; 3 = non-binary
gender = data[:, 216]
# age
age = data[:, 215]
# demographic? age, gender
demographic = data[:, 215:217]

# use pandas to remove nan values
# put all data we want together to remove corresponding rows with nan values
allQ = np.concatenate((artRatings, energy, demographic), axis=1)
allClean = pd.DataFrame(allQ).dropna().to_numpy()

# new indices
newArt = allClean[:, :91]
newEnergy = allClean[:, 92:183]
newDemog = allClean[:, 182:]


# to have same size data for linear regression:
# row means: because we want mean art rating per participant
energyMean = np.mean(newEnergy, axis=1).reshape(279, 1)  # x

demogMean = np.mean(newDemog, axis=1)

# energy mean + demographic
# because taking mean of age or gender doesn't really make sense? (maybe for age)
energyDemographic = np.concatenate((energyMean, newDemog), axis=1)

# to have same size data for linear regression:
# row medians: because we want median art rating per participant
# totalMean = np.mean(totalClean, axis = 1) # x
artMean = np.mean(newArt, axis=1)  # y


# cross-validation
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(
    energyDemographic, artMean, test_size=0.5, random_state=16654133)

# build the linear regression model using training data
model2 = LinearRegression().fit(xTrain2, yTrain2)
b0_2, b1_2 = model2.coef_, model2.intercept_


# cross-validation: using the betas from the training dataset, but
# measuring the error with the test dataset
yHat2 = b0_2[0] * xTrain2[:, 0] + b0_2[1] * \
    xTrain2[:, 1] + b0_2[2] * xTrain2[:, 2] + b1_2

#yHat = b1[0]*x[:,0] + b1[1]*x[:,1] + b1[2]*x[:,2] + b1[3]*x[:,3] + b1[4]*x[:,4] + b1[5]*x[:,5] + b1[6]*x[:,6] + b0

yPred2 = b0_2[0] * xTest2[:, 0] + b0_2[1] * \
    xTest2[:, 1] + b0_2[2] * xTest2[:, 2] + b1_2
# rmse and R^2
rmse2 = np.sqrt(np.mean((yPred2 - yTest2)**2))
rSqr2 = model2.score(energyDemographic, artMean)


# scatter plot between predicted and actual score of full model:
#plt.plot(xTest2, yTest2, "o", markersize = 4)
#plt.scatter(yPred2, yTest2)
plt.plot(yTest2, yPred2, "o", markersize=4)
plt.ylabel('Predicted Art Rating')
plt.xlabel('Actual Art Rating')
plt.suptitle("Art Preference vs. Energy Ratings and Demographic Info")
plt.title(f'RMSE = {rmse2:.3f}, R^2 = {rSqr2:.3f}', fontsize=10)
#plt.title('R^2 = {:.3f}'.format(rSqr2))
plt.show()


# %% Question 7: Considering the 2D space of average preference ratings vs. average energy rating (that
# contains the 91 art pieces as elements), how many clusters can you – algorithmically - identify
# in this space? Make sure to comment on the identity of the clusters – do they correspond to
# particular types of art?

artRatings = data[:, :91]
energy = data[:, 91:182]

artEnergy = np.concatenate([artRatings, energy], axis=1)
artEnergyClean = pd.DataFrame(artEnergy).dropna().to_numpy()

artClean = artEnergyClean[:, :91]
energyClean = artEnergyClean[:, 91:182]

artCleanMean = np.mean(artClean, axis=0)
energyCleanMean = np.mean(energyClean, axis=0)


x_new = np.column_stack((artCleanMean, energyCleanMean))


# SILHOUETTE TO FIND IDEAL k VALUE
# Init:
numClusters = 9  # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters, 1])*np.NaN  # init container to store sums

# Compute kMeans for each k:
for ii in range(2, numClusters+2):  # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters=int(ii)).fit(
        x_new)  # compute kmeans using scikit
    cId = kMeans.labels_  # vector of cluster IDs that the row belongs to
    # coordinate location for center of each cluster
    cCoords = kMeans.cluster_centers_
    # compute the mean silhouette coefficient of all samples
    s = silhouette_samples(x_new, cId)
    sSum[ii-2] = sum(s)  # take the sum
    # Plot data:
    plt.subplot(3, 3, ii-1)
    plt.hist(s, bins=20)
    plt.xlim(-0.2, 1)
    plt.ylim(0, 20)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    # sum rounded to nearest integer
    plt.title('Sum: {}'.format(int(sSum[ii-2])), fontsize=10)
    plt.tight_layout()  # adjusts subplot
plt.show()

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2, numClusters, 9), sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

# kMeans yields the coordinates centroids of the clusters, given a certain number k
# of clusters. Silhouette yields the number k that yields the most unambiguous clustering
# This number k is the maximum of the summed silhouette scores.

# NOW WE CLUSTER
# Now that we determined the optimal k, we can now ask kMeans to cluster the data for us,
# assuming that k

# kMeans:
numClusters = 4
kMeans = KMeans(n_clusters=numClusters).fit(x_new)
cId = kMeans.labels_
cCoords = kMeans.cluster_centers_

# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x_new[plotIndex, 0], x_new[plotIndex, 1], 'o', markersize=3)
    plt.plot(cCoords[int(ii-1), 0], cCoords[int(ii-1), 1],
             'o', markersize=5, color='black')
    plt.xlabel('Average Art Preference Ratings')
    plt.ylabel('Average Energy Ratings')
    plt.title("Energy vs Art Ratings")


# %% Question 8: Considering only the first principal component of the self-image ratings as
# inputs to a regression model – how well can you predict art preference ratings from that factor alone?

# self-image column questions (10 questions)
selfImage = data[:, 205:215]

artSelf = np.concatenate([artRatings, selfImage], axis=1)
artSelfClean = pd.DataFrame(artSelf).dropna().to_numpy()

artClean = artSelfClean[:, :91]

# mean art
artCleanMean = np.mean(artClean, axis=1).reshape(286, 1)

selfClean = artSelfClean[:, 91:]


# Compute correlation between each varible (question) across all courses:
corrMatrix2 = np.corrcoef(selfClean, rowvar=False)
# Column-wise is appropriate here because we want to know how similar questions were answered
# and questions are in the columns.

# Plot the correlation matrix:
plt.imshow(corrMatrix2)
plt.xlabel('Questions Self-Image')
plt.ylabel('Questions Self-Image')
plt.colorbar()
plt.show()


# PCA: want to find the first principal component: the type of question that can
# encapsulate others well

# 1. Z-score the data:
zscoredData2 = stats.zscore(selfClean)

# 2. Initialize PCA object and fit to our data:
pca2 = PCA().fit(zscoredData2)

# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals2 = pca2.explained_variance_
# If we order the eigenvectors in decreasing order of eigenvalue = "Principal components"

# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
loadings2 = pca2.components_  # Rows: Eigenvectors. Columns: Where they are pointing
# In other words, not mean centered, not-z-scored data will yield nonsense, if fed to a PCA.

# 3c. Rotated Data: Simply the transformed data - people's ratings (rows) in
# terms of 10 questions variables (columns) ordered by decreasing eigenvalue
# (principal components)
rotatedData2 = pca2.fit_transform(zscoredData2)

# 4. For the purposes of this, you can think of eigenvalues in terms of
# variance explained:
varExplained2 = eigVals2/sum(eigVals2)*100

# Now let's display this for each factor:
for ii in range(len(varExplained2)):
    print(varExplained2[ii].round(3))


# Scree Plot: bar graph of the sorted Eigenvalues
x2 = np.linspace(1, 10, 10)
plt.bar(x2, eigVals2, color='gray')
# Orange Kaiser criterion line for the fox
plt.plot([0, 10], [1, 1], color='orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()


# plot based on first principle component: which accounts for 43.137% of the variance
# Select and look at one factor at a time, in Python indexing
whichPrincipalComponent = 0
# note: eigVecs multiplied by -1 because the direction is arbitrary
plt.bar(x2, loadings2[whichPrincipalComponent, :]*-1)
# and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.title("Good Self-Esteem?")
plt.show()  # Show bar plot

# pretty much all the questions are similar (question 4 a bit less but still pretty good)
# we could encapsulate all the questions under something like:
# "on the whole, i am satisfied/happy with myself // have good self-esteem"


# principle component 1
pc1 = rotatedData2[:, 0].reshape(286, 1)

# cross-validation
xTrain3, xTest3, yTrain3, yTest3 = train_test_split(
    pc1, artCleanMean, test_size=0.5, random_state=16654133)

# build the linear regression model using training data
model3 = LinearRegression().fit(xTrain3, yTrain3)
b0_3, b1_3 = model3.coef_, model3.intercept_


# cross-validation: using the betas from the training dataset, but
# measuring the error with the test dataset
yHat3 = b0_3 * xTrain3 + b1_3
yPred3 = b0_3 * xTest3 + b1_3
# yPred2 = model1.predict(xTest)
# rmse and R^2
rmse = np.sqrt(np.mean((yPred3 - yTest3)**2))
rSqr = model1.score(energyMean, artMean)


# scatter plot between predicted and actual score of full model:
'''
plt.scatter(yPred3, yTest3)
plt.xlabel('Predicted Art Rating')
plt.ylabel('Actual Art Rating')
'''
plt.plot(yTest3, yPred3, "o", markersize=4)
plt.ylabel('Predicted Art Rating')
plt.xlabel('Actual Art Rating')
plt.suptitle('Art Rating Based on Level of Good Self-Esteem')
plt.title(f'RMSE = {rmse:.3f}, R^2 = {rSqr:.3f}', fontsize=10)
plt.show()


# %% Question 9: Consider the first 3 principal components of the “dark personality” traits –
# use these as inputs to a regression model to predict art preference ratings. Which of these
# components significantly predict art preference ratings? Comment on the likely identity of
# these factors (e.g. narcissism, manipulativeness, callousness, etc.).

dark = data[:, 182:194]

artDark = np.concatenate([artRatings, dark], axis=1)
artDarkClean = pd.DataFrame(artDark).dropna().to_numpy()

artRatClean = artDarkClean[:, :91]
darkClean = artDarkClean[:, 91:]

artRatMeanClean = np.mean(artRatClean, axis=1)


# Compute correlation between each varible (question) across all courses:
corrMatrix3 = np.corrcoef(darkClean, rowvar=False)
# Column-wise is appropriate here because we want to know how similar questions were answered
# and questions are in the columns.

# Plot the correlation matrix:
plt.imshow(corrMatrix3)
plt.xlabel('Questions Dark Personality')
plt.ylabel('Questions Dark Personality')
plt.colorbar()
plt.show()


# PCA: want to find the first principal component: the type of question that can
# encapsulate others well

# 1. Z-score the data:
zscoredData3 = stats.zscore(darkClean)

# 2. Initialize PCA object and fit to our data:
pca3 = PCA().fit(zscoredData3)

# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals3 = pca3.explained_variance_
# If we order the eigenvectors in decreasing order of eigenvalue = "Principal components"

# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
loadings3 = pca3.components_  # Rows: Eigenvectors. Columns: Where they are pointing
# In other words, not mean centered, not-z-scored data will yield nonsense, if fed to a PCA.

# 3c. Rotated Data: Simply the transformed data - people's ratings (rows) in
# terms of 10 questions variables (columns) ordered by decreasing eigenvalue
# (principal components)
rotatedData3 = pca3.fit_transform(zscoredData3)

# 4. For the purposes of this, you can think of eigenvalues in terms of
# variance explained:
varExplained3 = eigVals3/sum(eigVals3)*100

# Now let's display this for each factor:
for ii in range(len(varExplained3)):
    print(varExplained3[ii].round(3))


# Scree Plot: bar graph of the sorted Eigenvalues
x3 = np.linspace(1, 12, 12)
plt.bar(x3, eigVals3, color='gray')
# Orange Kaiser criterion line for the fox
plt.plot([0, 12], [1, 1], color='orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()


# plot based on FIRST principle component
# Select and look at one factor at a time, in Python indexing
whichPrincipalComponent = 0
# note: eigVecs multiplied by -1 because the direction is arbitrary
plt.bar(x3, loadings3[whichPrincipalComponent, :]*-1)
# and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.title("Manipulativeness")
plt.show()  # Show bar plot

# plot based on SECOND principle component
# Select and look at one factor at a time, in Python indexing
whichPrincipalComponent = 1
# note: eigVecs multiplied by -1 because the direction is arbitrary
plt.bar(x3, loadings3[whichPrincipalComponent, :]*-1)
# and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.title("Narcissism")
plt.show()  # Show bar plot

# plot based on THIRD principle component
# Select and look at one factor at a time, in Python indexing
whichPrincipalComponent = 2
# note: eigVecs multiplied by -1 because the direction is arbitrary
plt.bar(x3, loadings3[whichPrincipalComponent, :]*-1)
# and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.title("Callousness")
plt.show()  # Show bar plot


# % Visualize our data in the new coordinate system:
plt.plot(rotatedData3[:, 0]*-1, rotatedData3[:, 1]*-1, 'o', markersize=1)
plt.xlabel('Manipulativeness')
plt.ylabel('Narcissism')
plt.show()

# % Visualize our data in the new coordinate system:
plt.plot(rotatedData3[:, 1]*-1, rotatedData3[:, 2]*-1, 'o', markersize=1)
plt.xlabel('Narcissism')
plt.ylabel('Callousness')
plt.show()

# % Visualize our data in the new coordinate system:
plt.plot(rotatedData3[:, 0]*-1, rotatedData3[:, 2]*-1, 'o', markersize=1)
plt.xlabel('Manipulativeness')
plt.ylabel('Callousness')
plt.show()


# principle component 1,2,3
pc3 = rotatedData3[:, 0].reshape(284, 1)

# cross-validation
xTrain4, xTest4, yTrain4, yTest4 = train_test_split(
    pc3, artRatMeanClean, test_size=0.5, random_state=16654133)

# build the linear regression model using training data
model4 = LinearRegression().fit(xTrain4, yTrain4)
b0_4, b1_4 = model4.coef_, model4.intercept_


# cross-validation: using the betas from the training dataset, but
# measuring the error with the test dataset
yHat4 = b0_4 * xTrain4 + b1_4
yPred4 = b0_4 * xTest4 + b1_4
# yPred2 = model1.predict(xTest)
# rmse and R^2
rmse = np.sqrt(np.mean((yPred4 - yTest4)**2))
rSqr = model1.score(pc3, artRatMeanClean)


'''
# scatter plot between predicted and actual score of full model:
plt.scatter(yPred4, yTest4)
plt.xlabel('Predicted Art Rating')
plt.ylabel('Actual Art Rating')
plt.title('RMSE = {:.3f}'.format(rmse))
plt.title('R^2 = {:.3f}'.format(rSqr))
plt.show()
'''

plt.plot(yTest4, yPred4, "o", markersize=4)
plt.ylabel('Predicted Art Rating')
plt.xlabel('Actual Art Rating')
plt.suptitle('Art Ratings Based on Dark Personality')
plt.title('RMSE = {:.3f}'.format(rmse))
plt.show()


# %% Question 10: Can you determine the political orientation of the users (to simplify things
# and avoid gross class imbalance issues, you can consider just 2 classes: “left” (progressive
# & liberal) vs. “non- left” (everyone else)) from all the other information available, using any
# classification model of your choice? Make sure to comment on the classification quality of this model.

dataClean = pd.DataFrame(data).dropna().to_numpy()

# Create the feature matrix X (art ratings) and target variable y (political orientation):
# X
dataNoPolitics = np.concatenate(
    [dataClean[:, :217], dataClean[:, 218:]], axis=1)

# y
# convert scale of 1-6 to binary "left" (0) vs "non-left" (1)
# Column 218: Political orientation (1 = progressive, 2 = liberal, 3 = moderate, 4 = conservative,
# 5 = libertarian, 6 = independent)
politics = dataClean[:, 217]

newPolitics = []
# left vs non left
for i in range(276):
    if politics[i] in [1, 2]:
        newPolitics.append(0)
    else:
        newPolitics.append(1)
newPolitics = np.array(newPolitics).reshape(276, 1)


# PCA dark personality (from earlier)
dark10 = dataClean[:, 182:194]
# 1. Z-score the data:
zscoredDataDark = stats.zscore(dark10)
# 2. Initialize PCA object and fit to our data:
pcaDark = PCA().fit(zscoredDataDark)
# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigValsDark = pcaDark.explained_variance_
# If we order the eigenvectors in decreasing order of eigenvalue = "Principal components"
# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
# Rows: Eigenvectors. Columns: Where they are pointing
loadingsDark = pcaDark.components_
# In other words, not mean centered, not-z-scored data will yield nonsense, if fed to a PCA.
# 3c. Rotated Data: Simply the transformed data - people's ratings (rows) in
# terms of 10 questions variables (columns) ordered by decreasing eigenvalue
# (principal components)
rotatedDataDark = pcaDark.fit_transform(zscoredDataDark)
# 4. For the purposes of this, you can think of eigenvalues in terms of
# variance explained:
varExplainedDark = eigValsDark/sum(eigValsDark)*100
# Now let's display this for each factor:
for ii in range(len(varExplainedDark)):
    print(varExplainedDark[ii].round(3))


# Scree Plot: bar graph of the sorted Eigenvalues
x5 = np.linspace(1, 12, 12)
plt.bar(x5, eigValsDark, color='gray')
# Orange Kaiser criterion line for the fox
plt.plot([0, 12], [1, 1], color='orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Dark Personality')
plt.show()

print()

# PCA action personalities
action10 = dataClean[:, 194:205]
# 1. Z-score the data:
zscoredDataAction = stats.zscore(action10)
# 2. Initialize PCA object and fit to our data:
pcaAction = PCA().fit(zscoredDataAction)
# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigValsAction = pcaAction.explained_variance_
# If we order the eigenvectors in decreasing order of eigenvalue = "Principal components"
# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
# Rows: Eigenvectors. Columns: Where they are pointing
loadingsAction = pcaAction.components_
# In other words, not mean centered, not-z-scored data will yield nonsense, if fed to a PCA.
# 3c. Rotated Data: Simply the transformed data - people's ratings (rows) in
# terms of 10 questions variables (columns) ordered by decreasing eigenvalue
# (principal components)
rotatedDataAction = pcaAction.fit_transform(zscoredDataAction)
# 4. For the purposes of this, you can think of eigenvalues in terms of
# variance explained:
varExplainedAction = eigValsAction/sum(eigValsAction)*100
# Now let's display this for each factor:
for ii in range(len(varExplainedAction)):
    print(varExplainedAction[ii].round(3))
    
# Scree Plot: bar graph of the sorted Eigenvalues
x5 = np.linspace(1, 11, 11)
plt.bar(x5, eigValsAction, color='gray')
# Orange Kaiser criterion line for the fox
plt.plot([0, 11], [1, 1], color='orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Action Preferences')
plt.show()


print()

# PCA self image (from earlier)
selfImage10 = dataClean[:, 205:215]
# 1. Z-score the data:
zscoredDataSelf = stats.zscore(selfImage10)
# 2. Initialize PCA object and fit to our data:
pcaSelf = PCA().fit(zscoredDataSelf)
# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigValsSelf = pcaSelf.explained_variance_
# If we order the eigenvectors in decreasing order of eigenvalue = "Principal components"
# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
# Rows: Eigenvectors. Columns: Where they are pointing
loadingsSelf = pcaSelf.components_
# In other words, not mean centered, not-z-scored data will yield nonsense, if fed to a PCA.
# 3c. Rotated Data: Simply the transformed data - people's ratings (rows) in
# terms of 10 questions variables (columns) ordered by decreasing eigenvalue
# (principal components)
rotatedDataSelf = pcaSelf.fit_transform(zscoredDataSelf)
# 4. For the purposes of this, you can think of eigenvalues in terms of
# variance explained:
varExplainedSelf = eigValsSelf/sum(eigValsSelf)*100
# Now let's display this for each factor:
for ii in range(len(varExplainedSelf)):
    print(varExplainedSelf[ii].round(3))

# Scree Plot: bar graph of the sorted Eigenvalues
x5 = np.linspace(1, 10, 10)
plt.bar(x5, eigValsSelf, color='gray')
# Orange Kaiser criterion line for the fox
plt.plot([0, 10], [1, 1], color='orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Self-Image')
plt.show()


# CONCATENATE THE PCAs WITH POLITICS, ART MEAN, ENERGY MEAN, AND REST OF DATA 
# AND PUT THAT AS INPUT OF X FOR LOG REGR.
# use the first three principle components of dark questions and action questions
# use the first two principle components of self image questions
mainDark = rotatedDataDark[:, :3]
mainAction = rotatedDataAction[:, :3]
mainSelf = rotatedDataSelf[:, :2]

# art mean + energy mean
artMean2 = np.mean(dataClean[:, :91], axis=1).reshape(276, 1)
energyMean2 = np.mean(dataClean[:, 91:182], axis=1).reshape(276, 1)


dataNoPolitics2 = np.concatenate([artMean2, energyMean2, mainDark, mainAction, mainSelf, dataNoPolitics[:, 215:]], axis=1)

# cross-validation
xTrain5, xTest5, yTrain5, yTest5 = train_test_split(dataNoPolitics2, newPolitics, test_size=0.5, random_state=16654133)


# Fit model:
model5 = LogisticRegression().fit(xTrain5, yTrain5)
slope5 = model5.coef_
yInt5 = model5.intercept_

accuracy = model5.score(xTest5, yTest5)
print(f"Accuracy of model: {round(accuracy,3)*100}%\n")

# Plot the model
# Format the data
x1 = np.linspace(1, 7, 138)
#y1 = x1 * slope5 + yInt5
y1 = model5.predict(xTrain5)
#y1 = model5.predict_proba(xTest5)[:, 1]
#sigmoid = expit(y1) # MANUALLY CODE HERE THE LOGIT FUNCTION TO SEE IF IT WORKS
#from math import e
#sigmoid = e**y1 / (1 - e**y1)
sigmoid = 1 / (1 + np.exp(-x1))


# Plot:
# the ravel function returns a flattened array
plt.plot(x1, sigmoid.ravel(), color='red', linewidth=3)

plt.scatter(xTest5[:,0], y1, color='black')
plt.hlines(0.5, 1, 7, colors='gray', linestyles='dotted')
plt.xlabel('Mean Art Ratings')
plt.xlim([1, 7])
plt.ylabel('Politcally Left or Non-Left?')
plt.yticks(np.array([0, 1]))
plt.show()



from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(yTest5, y1)
print(auc_score)


