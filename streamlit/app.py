import streamlit as st
import pickle
import pandas as pd


def main():
    style = """<div style='background-color:black; padding:12px'>
              <h1 style='color:white'>What is the estimated HDB resale price?</h1>
       </div>"""
    
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))

    hdb_age = left.number_input("Target HDB Age", min_value = 5, max_value = 99, step = 1, format='%d', value = 5),

    full_flat_type = st.selectbox("What flat type-model ?",
                                  ("1 ROOM Improved", "2 ROOM DBSS", "2 ROOM Improved", "2 ROOM Model A", 
                                   "2 ROOM Premium Apartment", "2 ROOM Standard",
                                  "3 ROOM DBSS", "3 ROOM - Improved","3 ROOM Model A", "3 ROOM New Generation",
                                  "3 ROOM Premium Apartment", "3 ROOM Simplified",
                                  "3 ROOM Standard", "3 ROOM Terrace",
                                  "4 ROOM Adjoined flat", "4 ROOM DBSS",
                                  "4 ROOM Improved", "4 ROOM Model A",
                                  "4 ROOM Model A2", "4 ROOM New Generation",
                                  "4 ROOM Premium Apartment", "4 ROOM Premium Apartment Loft",
                                  "4 ROOM Simplified","4 ROOM Standard","4 ROOM Terrace",
                                  "4 ROOM Type S1","5 ROOM Adjoined flat","5 ROOM DBSS",
                                  "5 ROOM Improved","5 ROOM Improved-Maisonette",
                                  "5 ROOM Model A","5 ROOM Model A-Maisonette",
                                  "5 ROOM Premium Apartment","5 ROOM Premium Apartment Loft",
                                  "5 ROOM Standard", "5 ROOM Type S2", "EXECUTIVE Adjoined flat",
                                  "EXECUTIVE Apartment","EXECUTIVE - Maisonette",
                                  "EXECUTIVE Premium Apartment",
                                  "EXECUTIVE Premium Maisonette",
                                  "MULTI-GENERATION Multi Generation"))
    

    mrt_nearest_distance = st.selectbox("Distance to MRT?", 
                                        ("A Stone's Throw Away(<5mins)", "Short Walk (5 to 10mins)", 
                                         "Short Bus Ride (10 to 15mins)", "Long Bus Ride (>20mins)"))

    mall_nearest_distance = st.selectbox("Distance to Mall?", 
                                        ("A Stone's Throw Away(<5mins)", "Short Walk (5 to 10mins)", 
                                         "Short Bus Ride (10 to 15mins)", "Long Bus Ride (>20mins)"))

    mid = st.selectbox("Floor Level?", 
                                 ("Down to Earth (1st to 4th storey)", "Middle Range (5th to 9th storey)", 
                                  "High Range (10th to 15th storey)", "Skyscraper (16th storey and above)"))    

    postal_sector = st.selectbox("Postal Sector?", 
                                 ("Raffles Place", 
                                  "Tanjong Pagar",
                                  "Queenstown",
                                  "Harbourfront",
                                  "Pasir Panjang",
                                  "Golden Mile",
                                  "Little India",
                                  "Bukit Timah",
                                  "Novena",
                                  "Toa Payoh",
                                  "Macpherson",
                                  "Geylang",
                                  "Katong",
                                  "Bedok",
                                  "Loyang",
                                  "Tampines",
                                  "Hougang",
                                  "Bishan",
                                  "Clementi Park",
                                  "Jurong",
                                  "Bukit Panjang",
                                  "Kranji",
                                  "Upper Thomson",
                                  "Yishun",
                                  "Seletar"))    

   
    button = st.button('Predict HDB Price')
    
    # if button is pressed
    if button:
        # make prediction
        result = predict_hdb_price(hdb_age,full_flat_type,mrt_nearest_distance,mall_nearest_distance,postal_sector,mid)
        st.success(f'The estimated HDB resale price is ${result:,}')


# load the train model
with open("HDB_model_final.pkl", 'rb') as lr:
    model = pickle.load(lr)


def predict_hdb_price(hdb_age,full_flat_type,mrt_nearest_distance,mall_nearest_distance,postal_sector,mid):
    
    # assigning specific values based on user input
    tranc_year = 2023
    
    if mrt_nearest_distance == "A Stone's Throw Away(<5mins)":
        mrt_nearest_distance = 400
    elif mrt_nearest_distance == "Short Walk (5 to 10mins)":
        mrt_nearest_distance = 800
    elif mrt_nearest_distance == "Short Bus Ride (10 to 15mins)":
        mrt_nearest_distance = 1200
    elif mrt_nearest_distance == "Long Bus Ride (>20mins)":
        mrt_nearest_distance = 1600   
    
    if mall_nearest_distance == "A Stone's Throw Away(<5mins)":
        mall_nearest_distance = 400
    elif mall_nearest_distance == "Short Walk (5 to 10mins)":
        mall_nearest_distance = 800
    elif mall_nearest_distance == "Short Bus Ride (10 to 15mins)":
        mall_nearest_distance = 1200
    elif mall_nearest_distance == "Long Bus Ride (>20mins)":
        mall_nearest_distance = 1600   

    if mid == "Down to Earth (1st to 4th storey)":
        mid = 2
    elif mid == "Middle Range (5th to 9th storey)":
        mid = 8
    elif mid == "High Range (10th to 15th storey)":
        mid = 12
    elif mid == "Skyscraper (16th storey and above)":
        mid = 40

    #importing the required tables
    df_sector_CBD = pd.read_csv("postal_sector_mean_dist_CBD.csv")
    df_flat_sqm = pd.read_csv("full_flat_type_mean_sqm.csv")
    df_sector_user = pd.read_csv("postal_sector_user_selection.csv", sep = ";")
    df = pd.read_csv("feature_names.csv")

    # vlookup the required values
    floor_area_sqm = df_flat_sqm.loc[df_flat_sqm["full_flat_type"] == full_flat_type]["mean_floor_area_sqm"].values[0]
    postal_sector = df_sector_user[df_sector_user["postal_sector_user"] == postal_sector]["postal_sector"].values[0]
    dist_CBD = df_sector_CBD[df_sector_CBD["postal_sector"] == postal_sector]["mean_dist_CBD"].values[0]

    # insert user values into the df
    df["tranc_year"] = tranc_year
    
    postal_sector_select = "postal_sector_" + str(postal_sector)
    df[postal_sector_select] = 1

    flat_select = "full_flat_type_" + full_flat_type
    df[flat_select] = 1

    df['mrt_nearest_distance'] = mrt_nearest_distance
    df['floor_area_sqm'] = floor_area_sqm
    df['mall_nearest_distance'] = mall_nearest_distance
    df["dist_CBD"] = dist_CBD
    df["mid"] = mid
    df["hdb_age"] = hdb_age

    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result

if __name__ == '__main__':
    main()
