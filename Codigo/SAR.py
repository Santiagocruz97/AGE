import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.interpolate import griddata
from libpysal.weights import Queen
from spreg import ML_Lag
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import contextily as ctx

# ---------------------------
# 1. CARGA DE DATOS
# ---------------------------
area_col = gpd.read_file("C:/Git_SCA/AGE/Area_Col").to_crs(epsg=4326)
depositos = gpd.read_file("C:/Git_SCA/AGE/Depositos").to_crs(epsg=4326)
corteza = gpd.read_file("C:/Git_SCA/AGE/Corteza").to_crs(epsg=4326)
anomalias = gpd.read_file("C:/Git_SCA/AGE/Anomalias_corregido").to_crs(epsg=4326)

# ---------------------------
# 2. INTERPOLACIÓN DEL MOHO EN UNA GRILLA
# ---------------------------
minx, miny, maxx, maxy = -80, -5, -65, 15
n_grid = 300
lon = np.linspace(minx, maxx, n_grid)
lat = np.linspace(miny, maxy, n_grid)
LON, LAT = np.meshgrid(lon, lat)

coords_m = np.vstack([corteza.geometry.x, corteza.geometry.y]).T
vals_m = corteza["Z"].values
grid_z = griddata(coords_m, vals_m, (LON, LAT), method="cubic")
grid_z[np.isnan(grid_z)] = griddata(coords_m, vals_m, (LON, LAT), method="nearest")[np.isnan(grid_z)]

# ---------------------------
# 3. ASIGNACIÓN DE VARIABLES A DEPÓSITOS
# ---------------------------
coords_dep = np.vstack([depositos.geometry.x, depositos.geometry.y]).T
anomalias_union = unary_union(anomalias.geometry)
depositos["moho"] = griddata((LON.flatten(), LAT.flatten()), grid_z.flatten(), (coords_dep[:, 0], coords_dep[:, 1]), method="cubic")
depositos["anomalia"] = depositos.geometry.within(anomalias_union).astype(int)
depositos["Presencia"] = 1
depositos = depositos.dropna(subset=["moho"])

# ---------------------------
# 4. GENERACIÓN DE PSEUDO-AUSENCIAS
# ---------------------------
union_col = unary_union(area_col.geometry)
rng = np.random.default_rng(42)
abs_points = []

while len(abs_points) < len(depositos):
    x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
    p = Point(x, y)
    if union_col.contains(p):
        abs_points.append(p)

gdf_abs = gpd.GeoDataFrame(geometry=abs_points, crs="EPSG:4326")
gdf_abs["moho"] = griddata((LON.flatten(), LAT.flatten()), grid_z.flatten(), (gdf_abs.geometry.x, gdf_abs.geometry.y), method="cubic")
gdf_abs["anomalia"] = gdf_abs.geometry.within(anomalias_union).astype(int)
gdf_abs["Presencia"] = 0
gdf_abs = gdf_abs.dropna(subset=["moho"])

# ---------------------------
# 5. UNIFICAR DATOS PARA EL MODELO
# ---------------------------
data = pd.concat([depositos, gdf_abs], ignore_index=True).reset_index(drop=True)

# ---------------------------
# 6. MATRIZ DE PESOS ESPACIALES (REINA)
# ---------------------------
w = Queen.from_dataframe(data)
w.transform = 'r'

# ---------------------------
# 7. VARIABLES INDEPENDIENTES Y DEPENDIENTES
# ---------------------------
X = data[["moho", "anomalia"]].values
y = data["Presencia"].values.reshape(-1, 1)

# ---------------------------
# 8. ESCALAR VARIABLES
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 9. MODELO ESPACIAL SAR (Lag)
# ---------------------------
model = ML_Lag(y, X_scaled, w=w, name_y="Presencia", name_x=["moho", "anomalia"], name_w="Queen")
print(model.summary)

# ---------------------------
# 10. MAPA DE PROBABILIDAD PREDICHA
# ---------------------------
# Evaluar presencia de anomalía sobre la grilla
anomalia_grid = np.array([
    int(Point(x, y).within(anomalias_union))
    for x, y in zip(LON.flatten(), LAT.flatten())
])

X_grid = np.column_stack([grid_z.flatten(), anomalia_grid])
X_grid_scaled = scaler.transform(X_grid)

# Calcular predicciones (producto punto + intercepto)
y_pred_sar = (X_grid_scaled @ model.betas[1:]) + model.betas[0]

# Crear GeoDataFrame con predicciones
pred_coords = [Point(x, y) for x, y in zip(LON.flatten(), LAT.flatten())]
gdf_pred = gpd.GeoDataFrame({"prob_sar": y_pred_sar.flatten()}, geometry=pred_coords, crs="EPSG:4326")
gdf_pred = gdf_pred.to_crs(epsg=3857)

# Proyección de capas
depositos_web = depositos.to_crs(epsg=3857)
gdf_abs_web = gdf_abs.to_crs(epsg=3857)
area_col_web = area_col.to_crs(epsg=3857)
anomalias_web = anomalias.to_crs(epsg=3857)

# ---------------------------
# 11. VISUALIZACIÓN DEL MAPA
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 10))
gdf_pred.plot(column="prob_sar", ax=ax, cmap="plasma", alpha=0.7, markersize=5, legend=True)
area_col_web.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
anomalias_web.boundary.plot(ax=ax, edgecolor="blue", linestyle="--", linewidth=0.8)
depositos_web.plot(ax=ax, color="black", markersize=25, label="Depósitos")
gdf_abs_web.plot(ax=ax, color="white", edgecolor="black", markersize=8, label="Pseudo-ausencias")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title("Mapa de probabilidad de depósitos - Modelo SAR", fontsize=14, pad=12)
ax.set_axis_off()
ax.legend(loc="lower right", title="Datos", fontsize=9)
plt.tight_layout()
plt.show()
