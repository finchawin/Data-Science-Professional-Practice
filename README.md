# Football Machine Learning Project: (Data Science Professional Practice)

## Executive Summary
Historically, football predcitions have relied on subjective analysis by journalists and pundits, but recently more and more studies have incorporated Machine Learning algorithms to predict match outcomes and final standings of leagues and tournaments.

This Study utilises match-level data sourced from ["Footballdata.co.uk"](https://www.football-data.co.uk/) to assess which mid-season features best predict final points.
<!-- This is a comment -->

## Data Preprocessing
Data was extracted for Premier League Seasons from 2015/16 - 2024/25 using the `pd.read_csv()` function and then concatenated into a single dataframe using `pd.concat()`.
Column consistency checks were performed to ensure that there would not be two columns that mean the same thing after concatenation. 

The `len()` method was then utilised to ensure each season had 380 rows (The number of games in a premier league season).

![Screenshot: Row checks](images/row_checks.png) 

### Building Team centric table
The dataframe was then transfomred into a team-centric format, creating two dataframes using `pd.DataFrame` one for home and one for away. A dictionary was used to map points (3 for a win , 1 for a draw and 0 for a loss) and columns were renamed to reduce ambguity.
![Screenshot: Team centric](images/team_centric.png) 

Match numbers were then assigned so the data could be filtered to 19 matches played (midseason) using a `pd.groupby().cumcount() +1`.
The dataframe was then filtered to 19 matches played and midseason statistics were calculated using a `pd.groupby().agg('sum')`. Sanity checks were conducted to ensure there were 38 games per team per season.
![Screenshot: filtering to 19_mp](images/filtering_19.png) 
![Screenshot: Group by](images/group_by.png) 

### Feature engineering
Additional features were engineered such as goal difference (Goals Scored - Goals Conceded) and a form index which calculated a teams form on a scale of 0-1 based off of its last 5 fixtures 

```python
# filter to first half of the season
mid_matches = team_match_long[team_match_long["match_number"] <= 19].copy()

# define function to calculate form (sum of last 5 games points / 15)
def form_last5(x):
    last5 = x.sort_values("match_number").tail(5)
    return last5["points"].sum() / 15  

# apply per Season–Team group
form_df = (
    mid_matches.groupby(["Season","team"])
               .apply(form_last5)
               .reset_index(name="form_index_mid")
)

# merge with mid_results 
mid_results = mid_results.merge(form_df, on=["Season","team"], how="left")
```
The data was then ready for EDA.
## EDA and Feature Selection
Histogram plots were used to assess the distribution of mid-season features and ensure their validity.

![Screenshot: Histograms](images/Histograms.png) 

Whilst a correlation barplot was used to assess which features most strongly correlated with final points. Correlation between features was also assessed to reduce the implications of multicollinearity and remove redundant features that  would have let to overfitting.

![Screenshot: correlation_plot](images/correlation_plot.png) 

This lead to the selection of the final machine learning ready model dataframe:

**Features:** **points_mid, goal_diff_mid, shots_target_mid, goals_against_mid, form_index**.

**Target Variable:** **Final Points**

![Screenshot: feature_selection](images/feature_selection.png) 


## Machine Learning

### Train and test split.
The model was then split using sci-kitlearn. Of the 200 rows (10 seasons 20 teams), **80%  was for training and 20% for testing**

```python
#Splitting into X and Y (features and target)
x = model_df[feature_cols]
y = model_df[target_col]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#train test split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=1234)
print("Train size:", x_train.shape[0], "rows")
print("Test size:", x_test.shape[0], "rows")
```

