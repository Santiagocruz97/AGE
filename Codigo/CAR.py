import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from libpysal.weights import Queen
from spreg import ML_Error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import contextily as ctx

# ---------------------------
# 1. CARGA DE DATOS
# ---------------------------
gdf_moho = gpd.read_file("C:/Git_SCA/AGE/Corteza").to_crs(epsg=4326)
gdf_depositos = gpd.read_file("C:/Git_SCA/AGE/Depositos").to_crs(epsg=4326)
gdf_anomalias = gpd.read_file("C:/Git_SCA/AGE/Anomalias_corregido").to_crs(epsg=4326)
gdf_area = gpd.read_file("C:/Git_SCA/AGE/Area_Col").to_crs(epsg=4326)

# ---------------------------
# 2. VARIABLE DEPENDIENTE: PRESENCIA DE DEPÓSITO
# ---------------------------
gdf_moho['Presencia'] = gdf_moho.sjoin(
    gdf_depositos[['geometry']], how='left', predicate='intersects'
)['index_right'].notnull().astype(int)

# ---------------------------
# 3. VARIABLE INDEPENDIENTE: ANOMALÍA GEOQUÍMICA
# ---------------------------
anomalias_union = unary_union(gdf_anomalias.geometry)
gdf_moho['anomalia'] = gdf_moho.geometry.within(anomalias_union).astype(int)

# ---------------------------
# 4. VARIABLES PARA EL MODELO
# ---------------------------
y = gdf_moho['Presencia'].values.reshape(-1, 1)
X_raw = gdf_moho[['Z', 'anomalia']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values)

# ---------------------------
# 5. MATRIZ DE PESOS ESPACIALES
# ---------------------------
w = Queen.from_dataframe(gdf_moho)
w.transform = 'r'

# ---------------------------
# 6. AJUSTAR MODELO CAR
# ---------------------------
model_car = ML_Error(
    y, X_scaled, w=w,
    name_y='Presencia',
    name_x=['moho', 'anomalia'],
    method='full'
)

print(model_car.summary)

# ---------------------------
# 7. AÑADIR PREDICCIONES AL GDF
# ---------------------------
gdf_moho["prob_car"] = model_car.predy.flatten()

# ---------------------------
# 8. VISUALIZACIÓN DEL MAPA
# ---------------------------
# Reproyectar capas a EPSG:3857
gdf_moho_web = gdf_moho.to_crs(epsg=3857)
gdf_depositos_web = gdf_depositos.to_crs(epsg=3857)
gdf_anomalias_web = gdf_anomalias.to_crs(epsg=3857)
gdf_area_web = gdf_area.to_crs(epsg=3857)

# Crear figura
fig, ax = plt.subplots(figsize=(10, 10))

# Mapa de probabilidad predicha
gdf_moho_web.plot(column="prob_car", ax=ax, cmap="plasma", markersize=20, legend=True, alpha=0.8)

# Límites del área y anomalías
gdf_area_web.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
gdf_anomalias_web.boundary.plot(ax=ax, edgecolor="blue", linestyle="--", linewidth=0.8)

# Depósitos reales
gdf_depositos_web.plot(ax=ax, color="black", markersize=25, label="Depósitos")

# Mapa base
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Estética
ax.set_title("Mapa de probabilidad de depósitos - Modelo CAR", fontsize=14, pad=12)
ax.set_axis_off()
ax.legend(loc="lower right", title="Datos", fontsize=9)
plt.tight_layout()
plt.show()
