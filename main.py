import pandas as pd

from ast import literal_eval

df = pd.read_csv("better_coords.csv")
df['polygon'] = df['polygon'].apply(literal_eval)
df["fill_color"] = df["fill_color"].apply(literal_eval)


score_to_status = {
    0: "Niski",
    1: "Normalny",
    2: "Wysoki",
    3: "Bardzo wysoki",
}

df["score_name"] = df["score"].apply(lambda x: score_to_status[x])

import streamlit as st
from geopy.geocoders import Nominatim
import pydeck as pdk

city_name = st.text_input("Enter a city in Poland:", "Warsaw")

# Step 2: Geocode it
geolocator = Nominatim(user_agent="gsm-access-app")
location = geolocator.geocode(f"{city_name}, Poland")

if location:
    st.success(f"Found: {location.address} ({location.latitude}, {location.longitude})")
    view_lat = location.latitude
    view_lon = location.longitude
    zoom_level = 10  # adjust as needed
else:
    st.error("City not found. Defaulting to center of Poland.")
    view_lat = 52.0
    view_lon = 19.0
    zoom_level = 6.5

RADIUS = 0.3  # ~30 km in degrees
filtered_df = df[
    (df['longitude'] >= view_lon - RADIUS) &
    (df['longitude'] <= view_lon + RADIUS) &
    (df['latitude'] >= view_lat - RADIUS) &
    (df['latitude'] <= view_lat + RADIUS)
].copy().sort_values("score", ascending=False)

data = filtered_df.iloc[:5]  # 5 first rows

selected_index = st.sidebar.selectbox("Zoom to top result (by index):", ["Brak wyboru"] +
                                      [f"Wynik {i + 1}" for i in range(len(data.index))])

if selected_index != "Brak wyboru":
    selected_row = data.iloc[int(selected_index.split(" ")[-1]) - 1]

    view_lat = selected_row['latitude']
    view_lon = selected_row['longitude']
    zoom_level = 13  # Adjust for desired zoom

filtered_df = df[
    (df['longitude'] >= view_lon - RADIUS) &
    (df['longitude'] <= view_lon + RADIUS) &
    (df['latitude'] >= view_lat - RADIUS) &
    (df['latitude'] <= view_lat + RADIUS)
].copy().sort_values("score", ascending=False)

# Step 3: Streamlit interface
st.title("Gdzie postawić paczkomat")

st.write(f"Total points: {len(df)}")

poly_layer = pdk.Layer(
    "PolygonLayer",
    data=filtered_df,
    get_polygon="polygon",
    get_fill_color="fill_color",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=1,
    pickable=True,
    auto_highlight=True,
    stroked=True,
    opacity=0.02,
)

# highlight_layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=data.iloc[selected_index],
#     get_position=["longitude", "latitude"],
#     get_fill_color=[0, 0, 0, 200],  # Black with some opacity
#     get_radius=100,  # Adjust for visibility
#     pickable=False,
# )

view_state = pdk.ViewState(
    latitude=view_lat,
    longitude=view_lon,
    zoom=zoom_level,
)

deck = pdk.Deck(
    layers=[poly_layer],
    initial_view_state=view_state,
    tooltip={"text": "Potencjał: {score_name}"},
    map_style="mapbox://styles/mapbox/light-v9"
)

st.pydeck_chart(deck)

for num, (_, row) in enumerate(data.iterrows()):
    st.sidebar.markdown(f"### Wynik {num + 1}")
    st.sidebar.write(f"**Potencjał:** {row['score_name']}")
    st.sidebar.write(f"**Latitude:** {row['latitude']:.5f}")
    st.sidebar.write(f"**Longitude:** {row['longitude']:.5f}")
    st.sidebar.write("---")


clicked = st.button("Explain results")

if clicked:
    explanation = "bla bla bla"

    st.text(explanation)

