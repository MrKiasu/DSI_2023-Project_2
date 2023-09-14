# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Analysis and Prediction of HDB Resale Flat Price

### Problem Statement

With rising resale flat prices, affordable public housing has become a concern to new homeowners. According to an annual wellness survey conducted by OCBC bank, 56% of Singaporeans who are in their 20s are worried that they cannot afford a home. 

As many aspiring homeowners will be financing their homes using their CPF savings, knowing the price of the HDB resale flat that they wish to purchase can help them better plan their finances and CPF contributions. 

So, how can the CPF Board assist these potential homeowners?

This project explores the key features that affect HDB resale prices in Singapore using data science methodologies. The objective is to build and deploy a reliable HDB resale price calculator that is intuitive and easy for potential homeowners to use, to help them understand what could be the resale price of their desired flat based on the key features. 
---

### Data Used

Source: Kaggle (https://www.kaggle.com/competitions/dsi-sg-project-2-regression-challenge-hdb-price/data)
* [`train.csv`](../datasets/'train.csv'): Training data containing HDB resale price and 76 features
* [`test.csv`](../datasets/'test.csv'): Test data containing 76 features

---

### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**id**|*integer*|unique id for each transaction|
|**tranc_yearmonth**|*string*|year and month of the resale transaction, e.g. 2015-02|
|**town**|*string*|HDB township where the flat is located, e.g. BUKIT MERAH|
|**flat_type**|*string*|type of the resale flat unit, e.g. 3 ROOM|
|**storey_range**|*string*|floor level (range) of the resale flat unit, e.g. 07 TO 09|
|**floor_area_sqm**|*float*|floor area of the resale flat unit in square metres|
|**flat_model**|*string*|HDB model of the resale flat, e.g. Multi Generation|
|**lease_commence_date**|*integer*|commencement year of the flat unit's 99-year lease|
|**resale_price**|*float*|the property's sale price in Singapore dollars|
|**tranc_year**|*integer*|year of resale transaction|
|**tranc_month**|*integer*|month of resale transaction|
|**lower**|*integer*|lower value of storey_range|
|**upper**|*integer*|upper value of storey_range|
|**mid**|*integer*|middle value of storey_range|
|**full_flat_type**|*string*|combination of flat_type and flat_model|
|**hdb_age**|*integer*|number of years from lease_commence_date to present year|
|**max_floor_lvl**|*integer*|highest floor of the resale flat|
|**year_completed**|*integer*|year which construction was completed for resale flat||
|**postal**|*string*|postal code of the resale flat block|
|**latitude**|*float*|Latitude based on postal code|
|**longitude**|*float*|Longitude based on postal code|
|**mall_nearest_distance**|*float*|distance (in metres) to the nearest mall|
|**mall_within_500m**|*float*|number of malls within 500 metres|
|**mall_within_1km**|*float*|number of malls within 1 kilometre|
|**mall_within_2km**|*float*|number of malls within 2 kilometre|
|**hawker_nearest_distance**|*float*|distance (in metres) to the nearest hawker centre|
|**hawker_within_500m**|*float*|number of hawker centres within 500 metres|
|**hawker_within_1km**|*float*|number of hawker centres within 1 kilometre|
|**hawker_within_2km**|*float*|number of hawker centres within 2 kilometre|
|**hawker_food_stalls**|*integer*|number of hawker food stalls in the nearest hawker centre|
|**mrt_nearest_distance**|*float*|distance (in metres) to the nearest MRT station|
|**mrt_name**|*string*|name of the nearest MRT station|
|**mrt_latitude**|*float*|latitude (in decimal degrees) of the the nearest MRT station|
|**mrt_longitude**|*float*|longitude (in decimal degrees) of the nearest MRT station|
|**bus_stop_nearest_distance**|*float*|distance (in metres) to the nearest bus stop|
|**bus_stop_name**|*string*|name of the nearest bus stop|
|**bus_stop_latitude**|*float*|latitude (in decimal degrees) of the the nearest bus stop|
|**bus_stop_longitude**|*float*|longitude (in decimal degrees) of the nearest bus stop|
|**pri_sch_nearest_distance**|*float*|distance (in metres) to the nearest primary school|
|**pri_sch_name**|*string*|name of the nearest primary school|
|**pri_sch_latitude**|*float*|latitude (in decimal degrees) of the the nearest primary school|
|**pri_sch_longitude**|*float*|longitude (in decimal degrees) of the nearest primary school|
|**postal_sector**|*int64*|first two digits of postal code|
|**dist_CBD**|*float*|distance (in metres) to Raffles Place MRT|

---

### Notebook description

01_Cleaning_final.ipynb 

* [`01_Cleaning_final`](../code/'01_Cleaning_final.ipynb'): Problem statement, data cleaning of train dataset, and feature engineering
* [`02_EDA_final`](../code/'02_EDA_final.ipynb'): Exploratory data analysis of features and feature selection
* [`03_Modelling and Prediction_final`](../code/'03_Modelling and Prediction_final.ipynb'): Data modelling, conclusions and recommendations
* [`04_Kaggle_final`](../code/'04_Kaggle_final.ipynb'): Data cleaning of test dataset, prediction using fitted model, Kaggle score
---

### Conclusion

- A linear regression model is trained with 9 key features.
- These features include: transaction year, floor area, age of HDB, full flat type, nearest distance to MRT, nearest distance to mall, postal sector, distance to CBD, and mid storey.
- With this model, 89% of the variations in resale price can be explained by the features, with ±$48000 price difference.

---

### Recommendations

1. Improve prediction score
 - Explore other factors (e.g. impact of cooling measures on resale price)
 - Explore other machine learning models

2. User Testing
- Test and improve app usability with focus groups

3. Public Launch
- Campaign launch to educate housing finance planning with calculation.

---

