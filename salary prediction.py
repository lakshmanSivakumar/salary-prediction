#SALARY PREDICTION USING MULTI VARIATE LINEAR REGRESSION
#Author - Lakshman Sivakuamr
#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pplt

# read data into a dataFrame
df = pd.read_csv('employee_data.csv')

# shape of the DataFrame
print(df.shape)

#data pre-processing
df.drop(df.columns[[0,1]],axis=1,inplace=True)
data = pd.DataFrame(df,columns=['groups'],dtype=str)
groups = np.array(data,dtype=str).reshape(1000,1)
blood_group = [ ]

for i in range(0,len(groups)):
  if groups[i] == 'A':
    blood_group.append(0)
  elif groups[i] == 'B':
    blood_group.append(1)
  elif groups[i] == 'AB':
    blood_group.append(2)
  elif groups[i] == 'O':
    blood_group.append(3)

df['blood_group'] = np.array(blood_group,dtype=int).reshape(1000,1)
m = len(df)

#feature formation for the dataset through array
data = pd.DataFrame(df)

x0 = np.ones([1000]).reshape(1000,1)
data = pd.DataFrame(df,columns=['age'])

x1 = np.array(data,dtype=np.float64).reshape(1000,1)
data = pd.DataFrame(df,columns=['healthy_eating'])

x2 = np.array(data,dtype=np.float64).reshape(1000,1)
data = pd.DataFrame(df,columns=['active_lifestyle'])

x3= np.array(data,dtype=np.float64).reshape(1000,1)
data = pd.DataFrame(df,columns=['blood_group'])

x4= np.array(data,dtype=np.float64).reshape(1000,1)
x = np.c_[x0, x1, x2, x3, x4]

data = pd.DataFrame(df,columns=['salary'])
y = np.array(data,dtype=np.float64).reshape(1000,1)

#pre-processed dataset
pd.set_option('display.max_columns', None)
print(df)

#class for multi variable linear regression
class MutliVariableLinearRegression:
  def __init__(self, x, lr):
    self.alpha = lr
    self.er_li = [ ]
    self.theta = np.zeros([5,1])
    def hypothesis(self, x):
    return np.dot(x, self.theta)

  def cost_function(self, x, y):
    return (1/2*m) * (np.sum(np.square(self.hypothesis(x) - y)))

  def gradient_descent(self, x, y, i):
    j = 0
    k = 0
    while i < 50000:
      if j == 5:
        j = 0
      if k == 1000:
        k = 0
        self.theta[j] = self.theta[j] - (self.alpha/m)*np.sum(((self.hypothesis(x) - y))*(x[k,:][j]) )
        error = self.cost_function(x, y)
        self.er_li.append(error)
      i += 1
      j += 1
      k += 1
      self.plot_function()
      return self.theta

  def plot_function(self):
    plt.title("Cost function graph")
    plt.xlabel("no. of iterations")
    plt.ylabel("cost function")
    plt.plot(self.er_li)
    plt.show()

if __name__ == '__main__':
  #learning Rate
  lr = 1e-6
  i = 0
  j = 0
  #creatring Object for the class LinearRegression
  LR = MutliVariableLinearRegression(x, lr)
  #gradient descent
  t = LR.gradient_descent(x, y, i)
  print("Theta 0 = ",t[0])
  print("Theta 1 = ",t[1])
  print("Theta 2 = ",t[2])
  print("Theta 3 = ",t[3])
  print("Enter the blood group of the employee (A or B or AB or O) - ", end = "")
  blood_group = input()
  if blood_group == 'A':
    blood_group = 0
  elif blood_group == 'B':
    blood_group = 1
  elif blood_group == 'AB':
    blood_group = 2
  elif blood_group == 'O':
    blood_group = 3
  print("Enter the age of the employee - ", end = "")
  age = int(input())
  print("Enter the eating habit of the employee on a scale of 0 - 10 - ", end = "")
  healthy_eating = int(input())
  print("Enter the life style of the employee on a scale of 0 - 10 - ", end = "")
  active_lifestyle = int(input())
  salary = t[0] + t[1]*age + t[2]*healthy_eating + t[3]*active_lifestyle + t[4]*blood_group
  print("The predicted salary of the employee is: ",salary)
  #correlation matrix and heatmap
  corrmat = df.corr()
  f, ax = pplt.subplots(figsize=(12, 9))
  sns.heatmap(corrmat, vmax=.8, square=True)
