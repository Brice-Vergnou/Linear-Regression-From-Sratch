import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def mse(y_true,y_pred) -> float: 
    return np.mean((y_true - y_pred) ** 2) 

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2
class LinearRegression:
    def __init__(self,lr=0.01,n_iter=10000) -> None:
        self.lr = lr
        self.n_iter = n_iter
        
    def predict(self,X) -> np.array:
        
        return np.dot(self.weights , X.T) + self.bias 
        
    def fit(self,X,y) -> None:
        
        n_samples , n_features = X.shape # Returns a tuple ( Number or rows , Number of columns )
        self.weights = np.zeros(n_features) # if one feature : ax +b , if two : ax + bx + c....
        self.bias = 0 # The intercept
        
        # Gradient descent
        for i in range(self.n_iter):
            
            # compute prediction
            y_pred = np.dot(self.weights , X.T) + self.bias 
            
            # compute derivates
            dw = ( 1 / n_samples ) *  ( np.dot(-2 *X.T,(y-y_pred)))
            db = ( 1 / n_samples ) * ( -2 *  np.sum(y-y_pred))
            
            # update weights and bias 
            self.weights -= dw * self.lr
            self.bias -= db * self.lr
            
            if i % 2500 == 0 or i==0:
                print(f"Iter : {i}   ,   MSE : {mse(y,self.predict(X)):.2f}   ,  R2  : {r2_score(y,self.predict(X)):.4f}")
            
    


X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression(lr=0.00001, n_iter=400000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

mse = mse(y_test, predictions)
print("\nFinal MSE:", mse)

accu = r2_score(y_test, predictions)
print("Final Accuracy:", accu)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()