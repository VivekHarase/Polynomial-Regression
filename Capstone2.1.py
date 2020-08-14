#=====================================
#CAPSTONE Task 22
'''Polynomial regression of
the first 30 days of Covid-19 cases
in South Africa'''
#Vivek Harase
#=====================================
#Required imports
import pandas as pd #For the dataframe
import matplotlib.pyplot as plt #For data visualisation
from matplotlib import rcParams
from sklearn.model_selection import train_test_split #All Scikit learn models for machine learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Pandas read in the x and y data of the cases data into 2 arrays
data = pd.read_csv('sa covid_19.csv') #Data source, Wikipedia first 30 days of cases in SA: Mar-Apr 2020
x = data['x'].values
y = data['y'].values

#Splitting the data into training and testing using function train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #Random state is an that integer shuffles the data for training and testing

#rcParams for visual styling
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

#Reshape function to shape the data by adding 2D dimensionality to arrays
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

#Reordering the arrays to override the shuffle by the train_test_split function
y_train = y_train[x_train[:,0].argsort()]
x_train = x_train[x_train[:, 0].argsort()]

#Setting the degree of the polynomial
poly = PolynomialFeatures(degree=3)

#Transforming the data into a new matrix of the given degree
x_poly = poly.fit_transform(x_train)

#Training the model
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y_train)

#Plot of the line according to the visual styling
plt.title('Disease cases regressed on days')
plt.xlabel('Days')
plt.ylabel('Number of cases')
plt.plot(x_train, poly_reg.predict(x_poly), c='#FFA500', label='Polynomial regression line')
plt.scatter(x_test, y_test, c='#0000CD', label='Testing data')
plt.scatter(x_train, y_train, c='#008000', label='Training data')
plt.legend(loc="upper left")
plt.show()
