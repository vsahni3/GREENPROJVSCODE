from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

dfs = {
  'Portugal' : "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=18289299&single=true&output=csv",
  'Russia' : "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
  'Phillipines' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1345178492&single=true&output=csv',
  'Somalia' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1812151718&single=true&output=csv',
  'Pakistan' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1649849783&single=true&output=csv',
  'South Africa' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=916816526&single=true&output=csv',
  'Nigeria' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=2097142338&single=true&output=csv',
  'South Korea' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1268294931&single=true&output=csv',
  'Madagascar' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1381693534&single=true&output=csv',
  'Spain' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=2087802250&single=true&output=csv',
  'Kenya' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1731413971&single=true&output=csv',
  'Sweden' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1463763512&single=true&output=csv',
  'Japan' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=171652347&single=true&output=csv',
  'Switzerland' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1702770149&single=true&output=csv',
  'Italy' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1584210141&single=true&output=csv',
  'Tanzania' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1669991726&single=true&output=csv',
  'Iran' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1867374964&single=true&output=csv',
  'Thailand' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=350035388&single=true&output=csv',
  'India' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=971734650&single=true&output=csv',
  'Uganda' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=614084376&single=true&output=csv',
  'Hong Kong' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1272408342&single=true&output=csv',
  'Ukraine' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1080596110&single=true&output=csv',
  'Haiti' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1864643547&single=true&output=csv',
  'United Kingdom' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=805405508&single=true&output=csv',
  'Germany' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=960569026&single=true&output=csv',
  'Uruguay' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=376411358&single=true&output=csv',
  'France' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1626888339&single=true&output=csv',
  'USA' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=967991399&single=true&output=csv',
  'Ethiopia' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=548609678&single=true&output=csv',
  'Venezuela' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=439592444&single=true&output=csv',
  'Egypt' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=654916613&single=true&output=csv',
  'Ecuador' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1174414357&single=true&output=csv',
  'Dominican Republic' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=778777951&single=true&output=csv',
  'Colombia' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1911216266&single=true&output=csv',
  'Chile' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=729963991&single=true&output=csv',
  'Canada' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1204593914&single=true&output=csv',
  'Cambodia' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=577115064&single=true&output=csv',
  'Brazil' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=2101093255&single=true&output=csv',
  'Bosnia' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1540684784&single=true&output=csv',
  'Barbados' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1246824260&single=true&output=csv',
  'Australia' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1902318173&single=true&output=csv',
  'Argentina' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=1865508675&single=true&output=csv',
  'Algeria' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=533378518&single=true&output=csv',
  'Afghanistan' : 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSKq8QYnPFAppmuTEmiyE6uFTESmdaiy8ggVr2GsOnGwXlwdIAmuWW8R54LzVHB1oj5tIVW2_6o5F-v/pub?gid=0&single=true&output=csv'

}



def test(country):
  """"get error values for any country you want"""

  df = pd.read_csv(dfs[country])
  X = df['Year']
  y = df[country]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

  def errors_products():
    numerator = (X_train - X_train.mean()) * (y_train - y_train.mean())
    return numerator

  def squared_errors():
    denom = (X_train - X_train.mean()) ** 2
    return denom

  slope = errors_products().sum() / squared_errors().sum()
  intercept = y_train.mean() - slope * X_train.mean()


  predictions = X_test * slope + intercept

  predictions_reshaped = predictions.values.reshape(-1, 1)
  y_test_reshaped = y_test.values.reshape(-1, 1)


  r2 = r2_score(y_test_reshaped, predictions_reshaped)
  MSE = mean_squared_error(y_test_reshaped, predictions_reshaped)
  RMSE = mean_squared_error(y_test_reshaped, predictions_reshaped) ** 0.5
  MAE = mean_absolute_error(y_test_reshaped, predictions_reshaped)
  return r2, MSE, RMSE, MAE

def main_testing():
	for i in dfs:
		r2, MSE, RMSE, MAE = test(i)
		if r2 < 0.7:
			print(f'r squared error value for country {i} is too low')
		# the rest of the error values are relative and need to be seen on a case to case basis
		# there are no absolute thresholds to compare with like with r squared