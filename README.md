# Football Machine Learning Project: (Data Science Professional Practice)

## Executive Summary
Historically, football predcitions have relied on subjective analysis by journalists and pundits, but recently more and more studies have incorporated Machine Learning algorithms to predict match outcomes and final standings of leagues and tournaments.

This Study utilises match-level data sourced from ["Footballdata.co.uk"](https://www.football-data.co.uk/) to assess which mid-season features best predict final points.
<!-- This is a comment -->

## Data Preprocessing
Data was extracted for Premier League Seasons from 2015/16 - 2024/25 using the `pd.read_csv()` function and then concatenated into a single dataframe using `pd.concat()`.
Column consistency checks were performed to ensure that there would not be two columns that mean the same thing after concatenation. 

The `len()` method was then utilised to ensure each season had 380 rows (The number of games in a premier league season).




![Histogram](images/histogram-example-2.png) 

