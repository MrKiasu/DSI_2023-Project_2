import streamlit as st
import pickle
import pandas as pd


def main():
    style = """<div style='background-color:white; padding:12px'>
              <h1 style='color:black'>House Price Prediction App Hello World</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))
    
    #tranc_year = left.number_input("The year of when this prediction is for", step = 1.0, format="%d", value = 2023),
    #floor_area_sqm = right.number_input("Target Floor Area (sqm)",  step = 1.0, format='%d', value = 100),

    hdb_age = left.number_input("Target HDB Age", min_value = 5, step = 1, format='%d', value = 5),

    full_flat_type = st.selectbox("What flat type-model ?",
                                  ("1 ROOM - Improved", "2 ROOM - DBSS", "2 ROOM - Improved", "2 ROOM - Model A", 
                                   "2 ROOM - Premium Apartment", "2 ROOM - Standard",
                                  "3 ROOM - DBSS", "3 ROOM - Improved","3 ROOM - Model A", "3 ROOM - New Generation",
                                  "3 ROOM - Premium Apartment", "3 ROOM - Simplified",
                                  "3 ROOM - Standard", "3 ROOM - Terrace",
                                  "4 ROOM - Adjoined flat", "4 ROOM - DBSS",
                                  "4 ROOM - Improved", "4 ROOM - Model A",
                                  "4 ROOM - Model A2", "4 ROOM - New Generation",
                                  "4 ROOM - Premium Apartment", "4 ROOM - Premium Apartment Loft",
                                  "4 ROOM - Simplified","4 ROOM - Standard","4 ROOM - Terrace",
                                  "4 ROOM - Type S1","5 ROOM - Adjoined flat","5 ROOM - DBSS",
                                  "5 ROOM - Improved","5 ROOM - Improved-Maisonette",
                                  "5 ROOM - Model A","5 ROOM - Model A-Maisonette",
                                  "5 ROOM - Premium Apartment","5 ROOM - Premium Apartment Loft",
                                  "5 ROOM - Standard", "5 ROOM - Type S2", "EXECUTIVE - Adjoined flat",
                                  "EXECUTIVE - Apartment","EXECUTIVE - Maisonette",
                                  "EXECUTIVE - Premium Apartment",
                                  "EXECUTIVE - Premium Maisonette",
                                  "MULTI-GENERATION - Multi Generation"))
    

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
                                  "Beach Road",
                                  "Golden Mile",
                                  "Little India",
                                  "Orchard",
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
                                  "Tengah",
                                  "Kranji",
                                  "Upper Thomson",
                                  "Yishun",
                                  "Seletar"))    

    #dist_CBD = st.selectbox("Distance to CBD Area?", 
    #                             ("Walkable (<2km)", "15mins Bus/MRT Trip (2 to 5km)", 
    #                              "30mins Bus/MRT Trip (5 to 10km)", "1hr Bus/MRT Trip (10 to 20km)"))
    
    
    button = st.button('Predict')
    
    # if button is pressed
    if button:
        # make prediction
        result = predict(hdb_age, mid, full_flat_type, postal_sector, mrt_nearest_distance, mall_nearest_distance)
        st.success(f'The predicted HDB resale price is ${result}')


# load the train model
with open("D:\Documents\GitHub\DSI_2023-Project_2\code\HDB_model.pkl", 'rb') as rf:
    model = pickle.load(rf)


def predict(floor_area_sqm, hdb_age, mid, full_flat_type, dist_CBD):
    # processing user input

    lists = [ floor_area_sqm, hdb_age, mid, full_flat_type, dist_CBD]
    df = pd.DataFrame(lists).transpose()

    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result

if __name__ == '__main__':
    main()
