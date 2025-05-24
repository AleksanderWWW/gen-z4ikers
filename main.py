import pandas as pd

from ast import literal_eval

import streamlit as st
from geopy.geocoders import Nominatim
import pydeck as pdk

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOllama(
        model="gemma3:latest",
        temperature=0,
        streaming=True
    )

llm = get_llm()

def score_to_color(score):
    if score < 5:
        return [255, 255, 255, 180]  # white-ish
    if score < 50:
        return [255, 120, 120, 220]  # light red
    if score < 70:
        return [255, 60, 60, 230]   # medium red

    return [180, 0, 0, 200]      # dark red


df = pd.read_csv("Finale/final_df_with_opportunity_scores2.csv")
df['polygon'] = df['polygon'].apply(literal_eval)


def score_to_status(score: float):
    if score < 5:
        return "Niski"

    if score < 50:
        return "Średni"

    if score < 70:
        return "Wysoki"

    return "Bardzo wysoki"



city_name = st.text_input("Wprowadź nazwę miasta w Polsce:", "Warszawa")
provider = st.selectbox("Dostawca paczkomatów", [
    "Paczkomat InPost",
    "Allegro One Box",
    "DPD Pickup",
    "Orlen Paczka",
    "DHL BOX 24/7"
])

score_col = f"scaled_{provider}_opportunity_score"

df["score_name"] = df[score_col].apply(score_to_status)
df["fill_color"] = df[score_col].apply(score_to_color)

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
].copy().sort_values(score_col, ascending=False)

data = filtered_df.iloc[:5]  # 5 first rows

selected_index = st.sidebar.selectbox("Przybliż do najwyższego wyniku (po indeksie):", ["Brak wyboru"] +
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
].copy().sort_values(score_col, ascending=False)

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
    st.sidebar.write(f"**Surowy wynik:** {row[score_col]}")
    st.sidebar.write(f"**Latitude:** {row['latitude']:.5f}")
    st.sidebar.write(f"**Longitude:** {row['longitude']:.5f}")
    st.sidebar.write("---")

print(data.columns)


# Prepare structured feature importance information
feature_importance_text = """
Najważniejsze cechy wpływające na ocenę lokalizacji (wg ważności):

Kluczowe cechy (najwyższy wpływ):
- footway (chodniki)
- parking

Istotne cechy (średni wpływ):
- supermarket
- latitude (szerokość geograficzna)
- tram_stop (przystanki tramwajowe)
- atm (bankomaty)
- convenience (sklepy convenience)
- residential (obszary mieszkalne)
- living_street (strefy zamieszkania)
"""

if selected_index != "Brak wyboru":
    # Prepare prompt
    system_prompt = f"""Jesteś ekspertem ds. analizy lokalizacji, pomagającym użytkownikom zrozumieć potencjalne możliwości umieszczenia paczkomatów.

    Otrzymasz szczegółowe dane dotyczące obszaru siatki 2km x 2km wraz z oceną potencjału (w skali 0-100) wskazującą na możliwość umieszczenia nowego paczkomatu. Wyższe wyniki oznaczają lepsze lokalizacje.

    Wyjaśniając ocenę potencjału użytkownikom:
    1. Zacznij od jasnego, jednozdaniowego podsumowania ogólnego potencjału lokalizacji (doskonały, dobry, umiarkowany lub słaby)
    2. Podkreśl 3-4 najważniejsze czynniki wpływające na tę ocenę, koncentrując się na:
    - Istniejącej infrastrukturze (obecne paczkomaty, parkingi, transport publiczny)
    - Demografii (gęstość zaludnienia, dochody gospodarstw domowych)
    - Dostępności (główne drogi, ruch pieszy, dostęp całodobowy)
    - Czynnikach komercyjnych (punkty zainteresowania, kategoria zagospodarowania terenu)
    3. Przedstaw krótką, możliwą do realizacji rekomendację na podstawie danych

    Utrzymaj swoją odpowiedź zwięzłą (łącznie 3-5 zdań). Używaj konwersacyjnego języka, który będzie zrozumiały dla nietechnicznych odbiorców.

    Weź pod uwagę również ważność zmiennych z modelu:
    {feature_importance_text}
    """

    human_prompt = f"""Proszę wyjaśnij ocenę potencjału i kluczowe spostrzeżenia dla lokalizacji na siatce:
    
    Dane lokalizacji oznaczające liczbę poniższych punktów:
    - Footway (chodniki): {data.iloc[int(selected_index.split(" ")[-1]) - 1]['footway']:.0f}
    - Parking: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['parking']:.0f}
    - Supermarket: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['supermarket']:.0f}
    - Współrzędne: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['latitude']:.5f}, {data.iloc[int(selected_index.split(" ")[-1]) - 1]['longitude']:.5f}
    - Przystanki tramwajowe: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['tram_stop']:.0f}
    - Bankomaty: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['atm']:.0f}
    - Sklepy convenience: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['convenience']:.0f}
    - Obszary mieszkalne: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['residential']:.0f}
    - Strefy zamieszkania: {data.iloc[int(selected_index.split(" ")[-1]) - 1]['living_street']:.0f}
    
    Potencjał lokalizacji: {data.iloc[int(selected_index.split(" ")[-1]) - 1][score_col]:.2f}/100
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

# Run analysis button
if st.button("Wyjaśnij wyniki"):
    with st.spinner("Analyzing location data..."):
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in llm.stream(messages):
                if isinstance(chunk, AIMessage):
                    text_chunk = chunk.content
                    full_response += text_chunk
                    response_container.markdown(full_response)
