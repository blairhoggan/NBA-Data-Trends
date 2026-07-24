import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

#to make the output look nicer
np.set_printoptions(suppress=True, precision = 2)

# View data
nba = pd.read_csv('nba_games.csv')
nba.head()

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

# Compare Knicks to Nets with respect to points scored per game in 2010
knicks_pts = nba_2010[nba_2010.fran_id == 'Knicks']['pts']
nets_pts = nba_2010[nba_2010.fran_id == 'Nets']['pts']

# Calculated difference between both teams average points per game in 2010
diff_means_2010 = knicks_pts.mean() - nets_pts.mean()
print(diff_means_2010)

# Histogram to compare points scored between both teams
plt.hist(knicks_pts, alpha = 0.8, density = True, label = 'Knicks')
plt.hist(nets_pts, alpha = 0.8, density = True, label = 'Nets')
plt.title('2010 Season')
plt.legend()
plt.show

# Repeated steps but for the 2014 season
knicks_pts_2014 = nba_2014[nba_2014.fran_id == 'Knicks']['pts']
nets_pts_2014 = nba_2014[nba_2014.fran_id == 'Nets']['pts']
diff_means_2014 = knicks_pts_2014.mean() - nets_pts_2014.mean()
print(diff_means_2014)

plt.hist(knicks_pts_2014, alpha = 0.8, density = True, label = 'Knicks')
plt.hist(nets_pts_2014, alpha= 0.8, density = True, label = 'Nets')
plt.legend()
plt.title('2014 Season')
plt.show()

# Boxplot of points (y-axis) and team (x-axis) in the 2010 season
sns.boxplot(data=nba_2010, x = 'fran_id', y = 'pts')
plt.show()

# Contingency table to help determine if teams win more playing at Home rather than Away
location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)
print(location_result_freq)

# Converted frequency table to proportion table
location_result_proportions = location_result_freq/len(nba_2010)
print(location_result_proportions)

# Calculated expected contingency table and Chi-Square statistic
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print(expected)
print(chi2)

# Calculate covariance between forecast and point_diff
covariance = np.cov(nba_2010.forecast, nba_2010.point_diff)
print(covariance)

# Calculate correlation between forecast and point_diff
point_diff_forecast_corr = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print(point_diff_forecast_corr)

# Scatterplot of point_diff vs forecast
plt.clf()
plt.scatter(data=nba_2010, x='forecast', y='point_diff')
plt.xlabel('Win Probability')
plt.ylabel('Point Difference')
plt.show()






