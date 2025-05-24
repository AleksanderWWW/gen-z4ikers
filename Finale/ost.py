import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import ast
import folium
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium # Główny komponent do wyświetlania map Folium w Streamlit

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Mapa Paczkomatów Małopolska")
st.title("Interaktywna Mapa Paczkomatów i Danych OSM - Małopolska")

# --- Funkcje pomocnicze (aby kod był bardziej modularny) ---

@st.cache_data # Cache'owanie wczytywania danych, aby przyspieszyć ponowne uruchomienia
def load_shapefile(shp_file_path, category_key):
    """Wczytuje i upraszcza dane Shapefile."""
    if os.path.exists(shp_file_path):
        try:
            gdf = gpd.read_file(shp_file_path)
            
            # Upraszczanie geometrii
            if gdf.crs and 'cartesian' in str(gdf.crs.type_name).lower():
                tolerance = 20 # Zwiększona tolerancja dla lepszej wydajności
            else:
                tolerance = 0.0002 # Zwiększona tolerancja
            
            st.write(f"Upraszczanie geometrii dla {category_key} (oryginalnie {len(gdf)} obiektów)...")
            gdf.geometry = gdf.geometry.buffer(0)
            gdf.geometry = gdf.geometry.simplify(tolerance, preserve_topology=True)
            gdf = gdf[~gdf.geometry.is_empty].copy() # .copy() aby uniknąć SettingWithCopyWarning
            
            print(f"Wczytano i uproszczono: {category_key} - {len(gdf)} obiektów, CRS: {gdf.crs}")
            return gdf
        except Exception as e:
            st.error(f"Błąd podczas wczytywania {os.path.basename(shp_file_path)} dla {category_key}: {e}")
            return None
    else:
        st.warning(f"Plik nie istnieje dla kategorii {category_key}: {shp_file_path}")
        return None

@st.cache_data
def load_paczkomaty_data(csv_path):
    """Wczytuje i przetwarza dane o paczkomatach z pliku CSV."""
    try:
        paczkomaty_df = pd.read_csv(csv_path)
        if 'location' in paczkomaty_df.columns:
            def parse_location_string(loc_str):
                try:
                    if pd.isna(loc_str) or not isinstance(loc_str, str) or loc_str.strip() == "": return None, None
                    loc_dict = ast.literal_eval(loc_str)
                    if isinstance(loc_dict, dict): return loc_dict.get('longitude'), loc_dict.get('latitude')
                    else: return None, None
                except: return None, None
            
            parsed_coords = paczkomaty_df['location'].apply(lambda x: pd.Series(parse_location_string(x), index=['longitude_parsed', 'latitude_parsed']))
            paczkomaty_df['longitude'] = parsed_coords['longitude_parsed']
            paczkomaty_df['latitude'] = parsed_coords['latitude_parsed']
            
            paczkomaty_df.dropna(subset=['longitude', 'latitude'], inplace=True)
            if not paczkomaty_df.empty:
                paczkomaty_gdf = gpd.GeoDataFrame(
                    paczkomaty_df,
                    geometry=gpd.points_from_xy(paczkomaty_df.longitude, paczkomaty_df.latitude),
                    crs="EPSG:4326"
                )
                print(f"Utworzono GeoDataFrame dla paczkomatów: {len(paczkomaty_gdf)} obiektów.")
                return paczkomaty_gdf
            else:
                st.warning("Nie udało się sparsować żadnych współrzędnych paczkomatów.")
                return None
        else:
            st.error("Błąd: Brak kolumny 'location' w pliku paczkomaty.csv.")
            return None
    except FileNotFoundError:
        st.error(f"Błąd: Nie znaleziono pliku {csv_path}")
        return None
    except Exception as e:
        st.error(f"Błąd podczas wczytywania lub przetwarzania paczkomaty.csv: {e}")
        return None

def transform_crs(gdf, target_crs="EPSG:4326", layer_name="warstwa"):
    """Transformuje układ współrzędnych GeoDataFrame."""
    if gdf is None: return None
    if gdf.crs is None:
        st.warning(f"Brak CRS dla warstwy {layer_name}. Próba ustawienia na {target_crs}.")
        try:
            return gdf.set_crs(target_crs, allow_override=True)
        except Exception as e_crs:
            st.error(f"Nie udało się ustawić CRS dla {layer_name}: {e_crs}.")
            return None
    elif gdf.crs != target_crs:
        print(f"Transformuję CRS dla {layer_name} z {gdf.crs} do {target_crs}")
        return gdf.to_crs(target_crs)
    return gdf

# --- Główna logika aplikacji ---

# Ścieżki do danych
malopolskie_shp_folder = r"C:\Users\anton\OneDrive\Pulpit\Mastercard\zach"
paczkomaty_csv_path = r"C:\Users\anton\Downloads\paczkomaty.csv"

# Konfiguracja warstw OSM do wczytania i wyświetlenia
# Użytkownik będzie mógł wybrać, które warstwy chce widzieć
osm_layers_config = {
    "Drogi": {"file": "gis_osm_roads_free_1.shp", "style": lambda x: {'color': '#B0B0B0', 'weight': 1.5, 'opacity': 0.6}, "tooltip": ['fclass', 'name'], "sample_frac": 0.2, "default_show": True},
    "Linie kolejowe": {"file": "gis_osm_railways_free_1.shp", "style": lambda x: {'color': '#707070', 'weight': 1, 'opacity': 0.7, 'dashArray': '5, 5'}, "tooltip": ['fclass'], "default_show": True},
    "Wody (obszary)": {"file": "gis_osm_water_a_free_1.shp", "style": lambda x: {'fillColor': '#AACCFF', 'color': '#6699CC', 'weight': 0.5, 'fillOpacity': 0.5}, "tooltip": ['name'], "default_show": True},
    "Drogi wodne": {"file": "gis_osm_waterways_free_1.shp", "style": lambda x: {'color': '#6699CC', 'weight': 1, 'opacity': 0.7}, "tooltip": ['name'], "default_show": True},
    "Użycie terenu": {"file": "gis_osm_landuse_a_free_1.shp", "style": lambda x: {'fillColor': '#E1F0C4', 'color': '#B3D39C', 'weight': 0.5, 'fillOpacity': 0.3}, "tooltip": ['fclass'], "default_show": False},
    "Tereny naturalne": {"file": "gis_osm_natural_a_free_1.shp", "style": lambda x: {'fillColor': '#D4EFDF', 'color': '#A3D6B8', 'weight': 0.5, 'fillOpacity': 0.3}, "tooltip": ['fclass'], "default_show": False},
    "POI (punkty)": {"file": "gis_osm_pois_free_1.shp", "style": lambda x: {'fillColor':'darkgreen', 'color':'darkgreen', 'radius':3, 'weight':0.5, 'fillOpacity':0.7}, "tooltip": ['fclass', 'name'], "sample_frac": 0.1, "default_show": False, "is_point": True}, # is_point do specjalnego traktowania
    "Miejsca (punkty)": {"file": "gis_osm_places_free_1.shp", "style": lambda x: {'fillColor':'purple', 'color':'purple', 'radius':4, 'weight':0.5, 'fillOpacity':0.7}, "tooltip": ['fclass', 'name'], "sample_frac": 0.3, "default_show": False, "is_point": True},
    # "Budynki": {"file": "gis_osm_buildings_a_free_1.shp", "style": lambda x: {'fillColor': '#D3D3D3', 'color': '#A9A9A9', 'weight': 0.3, 'fillOpacity': 0.2}, "tooltip": ['type'], "sample_frac": 0.05, "default_show": False}, # Bardzo ciężka
}

# --- Sidebar z opcjami ---
st.sidebar.header("Opcje Wyświetlania Mapy")
selected_osm_layers = []
for layer_name, config in osm_layers_config.items():
    if st.sidebar.checkbox(layer_name, value=config.get("default_show", False)):
        selected_osm_layers.append(layer_name)

show_paczkomaty = st.sidebar.checkbox("Pokaż Paczkomaty", value=True)
use_marker_cluster = st.sidebar.checkbox("Grupuj markery Paczkomatów (MarkerCluster)", value=True)

# --- Wczytywanie danych ---
with st.spinner("Wczytywanie danych geoprzestrzennych... To może chwilę potrwać."):
    # Wczytaj paczkomaty
    paczkomaty_gdf = load_paczkomaty_data(paczkomaty_csv_path)
    paczkomaty_gdf_display = transform_crs(paczkomaty_gdf, layer_name="Paczkomaty")

    # Wczytaj wybrane warstwy OSM
    transformed_osm_data = {}
    for layer_name in selected_osm_layers:
        config = osm_layers_config[layer_name]
        shp_path = os.path.join(malopolskie_shp_folder, config["file"])
        gdf = load_shapefile(shp_path, layer_name)
        transformed_osm_data[layer_name] = transform_crs(gdf, layer_name=layer_name)

# --- Tworzenie mapy Folium ---
if paczkomaty_gdf_display is not None or any(transformed_osm_data.values()):
    st.subheader("Mapa")
    
    # Ustalenie środka i zoomu
    map_center = [50.0647, 19.9450] # Kraków
    zoom_start = 9

    if paczkomaty_gdf_display is not None and not paczkomaty_gdf_display.empty:
        center_geom = paczkomaty_gdf_display.unary_union.centroid
        map_center = [center_geom.y, center_geom.x]
    elif "Drogi" in transformed_osm_data and transformed_osm_data["Drogi"] is not None and not transformed_osm_data["Drogi"].empty:
        bounds = transformed_osm_data["Drogi"].total_bounds
        map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True)
    Fullscreen().add_to(m) # Przycisk pełnego ekranu

    # Dodawanie warstw OSM
    for layer_name, gdf in transformed_osm_data.items():
        if gdf is not None and not gdf.empty:
            config = osm_layers_config[layer_name]
            sample_frac = config.get("sample_frac", 1.0)
            
            gdf_to_display = gdf
            if sample_frac < 1.0 and len(gdf) * sample_frac > 1:
                st.write(f"Próbkowanie warstwy {layer_name} z frac={sample_frac}...")
                gdf_to_display = gdf.sample(frac=sample_frac, random_state=42)
            
            if gdf_to_display.empty: continue

            fg = folium.FeatureGroup(name=layer_name, show=config.get("default_show", False))
            
            if config.get("is_point", False): # Specjalne renderowanie dla punktów dla lepszej wydajności
                 for idx, row in gdf_to_display.iterrows():
                    tooltip_text = "<br>".join([f"{field.capitalize()}: {row.get(field, 'N/A')}" for field in config.get("tooltip", [])])
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=config["style"](None).get('radius', 3), # Pobierz radius ze stylu
                        color=config["style"](None).get('color', 'blue'),
                        fill=True,
                        fill_color=config["style"](None).get('fillColor', 'blue'),
                        fill_opacity=config["style"](None).get('fillOpacity', 0.6),
                        tooltip=tooltip_text if tooltip_text else None
                    ).add_to(fg)
            else: # Dla linii i poligonów
                folium.GeoJson(
                    gdf_to_display,
                    style_function=config["style"],
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=config.get("tooltip", []),
                        aliases=[f"{field.capitalize()}:" for field in config.get("tooltip", [])],
                        sticky=False
                    ) if config.get("tooltip") else None
                ).add_to(fg)
            fg.add_to(m)
            st.write(f"Dodano warstwę '{layer_name}' ({len(gdf_to_display)} obiektów) do mapy.")


    # Dodawanie paczkomatów
    if show_paczkomaty and paczkomaty_gdf_display is not None and not paczkomaty_gdf_display.empty:
        paczkomaty_fg = folium.FeatureGroup(name="Paczkomaty", show=True)
        
        target_for_markers = paczkomaty_fg
        if use_marker_cluster and len(paczkomaty_gdf_display) > 50: # Użyj klastra, jeśli > 50 paczkomatów
            st.write("Grupowanie markerów paczkomatów (MarkerCluster)...")
            marker_cluster = MarkerCluster(name="Paczkomaty (klastry)")
            marker_cluster.add_to(paczkomaty_fg)
            target_for_markers = marker_cluster
        
        for idx, row in paczkomaty_gdf_display.iterrows():
            popup_html = f"<b>{row.get('name', 'N/A')}</b><br>Status: {row.get('status', 'N/A')}<br><small>{row.get('address.line1', '')}</small>"
            if pd.notna(row.geometry.y) and pd.notna(row.geometry.x):
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=popup_html,
                    tooltip=row.get('name', 'Paczkomat'),
                    icon=folium.Icon(color='red', icon='cube', prefix='fa')
                ).add_to(target_for_markers)
        paczkomaty_fg.add_to(m)
        st.write(f"Dodano {len(paczkomaty_gdf_display)} paczkomatów do mapy.")

    folium.LayerControl(collapsed=False).add_to(m)

    # Wyświetlanie mapy w Streamlit
    with st.spinner("Renderowanie mapy..."):
        st_folium(m, width=None, height=600, returned_objects=[]) # None dla szerokości zajmie dostępną przestrzeń

else:
    st.warning("Brak danych do wyświetlenia na mapie. Sprawdź ścieżki do plików i wybrane warstwy.")

st.sidebar.info("Aplikacja do wizualizacji danych geoprzestrzennych dla Małopolski.")