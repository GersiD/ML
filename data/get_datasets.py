from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

# save as one the dataset
df = pd.concat([X, y], axis=1)
df.to_csv('iris.csv', index=False)

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 

# save as one the dataset
df = pd.concat([X, y], axis=1)
df.to_csv('wine_quality.csv', index=False)


