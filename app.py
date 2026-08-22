# ============================================================
# app.py — Dashboard TFM: Crédito Público Agropecuario Ecuador
# Ejecutar con: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from shapely.validation import make_valid
import requests
import io

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Crédito Agropecuario Ecuador",
    page_icon="🌾",
    layout="wide",               # usa todo el ancho de la pantalla
    initial_sidebar_state="expanded"
)

st.title("🌾 Crédito Público Agropecuario en Ecuador")
st.markdown(
    "**Dashboard de priorización territorial** basado en datos del MAG (2013-2025) "
    "y NBI del Censo 2022 (INEC). Permite identificar qué provincias tienen mayor "
    "brecha entre su nivel de pobreza y el crédito público recibido."
)

# ============================================================
# FUNCIONES DE CARGA DE DATOS (con caché para no recargar)
# @st.cache_data guarda el resultado en memoria:
# si los datos no cambian, no los vuelve a procesar
# ============================================================

@st.cache_data
def cargar_nbi():
    """Carga el dataset NBI por cantón (INEC Censo 2022)."""
    df = pd.read_csv("nbi_canton_2010_2022.csv")
    # Calculamos población total estimada por cantón
    df["pob_total_2022"] = (df["pobres_2022_n"] / df["pct_nbi_2022"]).round(0)
    return df

@st.cache_data
def cargar_geojson():
    """Carga y procesa el GeoJSON cantonal de Ecuador."""
    gdf_cant = gpd.read_file("ecuador.geojson")

    # Disolvemos cantones a nivel provincial
    gdf_prov = gdf_cant.dissolve(by="DPA_DESPRO", as_index=False)

    # Reparamos geometrías
    gdf_prov["geometry"] = gdf_prov["geometry"].apply(make_valid)

    # Mapeamos nombres de MAYÚSCULAS a formato estándar
    mapeo = {
        "AZUAY": "Azuay", "BOLIVAR": "Bolívar",
        "CAÐAR": "Cañar", "CAÑAR": "Cañar",
        "CARCHI": "Carchi", "CHIMBORAZO": "Chimborazo",
        "COTOPAXI": "Cotopaxi", "EL ORO": "El Oro",
        "ESMERALDAS": "Esmeraldas", "GALAPAGOS": "Galápagos",
        "GUAYAS": "Guayas", "IMBABURA": "Imbabura",
        "LOJA": "Loja", "LOS RIOS": "Los Ríos",
        "MANABI": "Manabí", "MORONA SANTIAGO": "Morona Santiago",
        "NAPO": "Napo", "ORELLANA": "Orellana",
        "PASTAZA": "Pastaza", "PICHINCHA": "Pichincha",
        "SANTA ELENA": "Santa Elena",
        "SANTO DOMINGO DE LOS TSACHILAS": "Santo Domingo De Los Tsáchilas",
        "SUCUMBIOS": "Sucumbíos", "TUNGURAHUA": "Tungurahua",
        "ZAMORA CHINCHIPE": "Zamora Chinchipe",
    }
    gdf_prov["province"] = gdf_prov["DPA_DESPRO"].map(mapeo)

    # Eliminamos zona no delimitada
    gdf_prov = gdf_prov[gdf_prov["province"].notna()].copy()

    # Calculamos puntos representativos para etiquetas
    def get_rep_point(row):
        if "GAL" in str(row["province"]).upper():
            return (-90.80, -0.95)
        try:
            rp = row.geometry.representative_point()
            return (rp.x, rp.y)
        except Exception:
            b = row.geometry.bounds
            return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    puntos = gdf_prov.apply(get_rep_point, axis=1)
    gdf_prov["rep_x"] = puntos.apply(lambda p: p[0])
    gdf_prov["rep_y"] = puntos.apply(lambda p: p[1])

    return gdf_prov

def procesar_mag(df_mag_raw, incluir_anio_incompleto=False):
    """
    Limpia y procesa el CSV del MAG.
    incluir_anio_incompleto=True: incluye todos los años (cuando el usuario sube un archivo)
    incluir_anio_incompleto=False: excluye el año incompleto (dataset por defecto)
    """
    df = df_mag_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Limpiamos el monto
    df["CP_VALOR_USD"] = (
        df["CP_VALOR_USD"].astype(str)
        .str.replace(",", "", regex=False).str.strip()
    )
    df["CP_VALOR_USD"] = pd.to_numeric(df["CP_VALOR_USD"], errors="coerce")

    # Meses en español
    meses = {
        "Enero":1,"Febrero":2,"Marzo":3,"Abril":4,
        "Mayo":5,"Junio":6,"Julio":7,"Agosto":8,
        "Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12
    }
    df["mes_num"] = df["CP_MES"].map(meses)
    df["CP_GENERO"] = df["CP_GENERO"].fillna("N/D")

    # Solo excluimos el año incompleto si estamos usando el dataset por defecto
    if not incluir_anio_incompleto:
        anio_max = df["CP_ANIO"].max()
        meses_anio_max = df[df["CP_ANIO"] == anio_max]["mes_num"].max()
        if meses_anio_max < 12:
            df = df[df["CP_ANIO"] < anio_max].copy()

    return df.dropna(subset=["CP_VALOR_USD"])

def calcular_ranking(df_mag, df_nbi):
    """
    Calcula el índice de brecha y ranking de priorización territorial.
    """
    # Agregamos MAG por provincia
    mag_prov = df_mag.groupby("DPA_DESPRO").agg(
        monto_total=("CP_VALOR_USD", "sum"),
        num_operaciones=("CP_NUM_OPERACIONES", "sum")
    ).reset_index()
    mag_prov["monto_millones"] = mag_prov["monto_total"] / 1_000_000
    mag_prov["promedio_op"] = (
        mag_prov["monto_total"] / mag_prov["num_operaciones"]
    ).round(0)

    # NBI provincial ponderado por población
    nbi_prov = df_nbi.groupby("provincia").apply(
        lambda g: pd.Series({
            "nbi_2022_pond": (
                (g["pct_nbi_2022"] * g["pob_total_2022"]).sum() /
                g["pob_total_2022"].sum()
            ).round(3)
        }),
        include_groups=False
    ).reset_index()

    # Población total por provincia
    pob_prov = df_nbi.groupby("provincia")["pob_total_2022"].sum().reset_index()
    pob_prov.columns = ["provincia", "pob_total"]

    # Cruce
    cruce = pd.merge(mag_prov, nbi_prov,
                     left_on="DPA_DESPRO", right_on="provincia", how="inner")
    cruce = pd.merge(cruce, pob_prov, on="provincia", how="left")
    cruce["credito_per_capita"] = (
        cruce["monto_total"] / cruce["pob_total"]
    ).round(2)

    # Índice de brecha
    scaler = MinMaxScaler()
    cruce["nbi_norm"] = scaler.fit_transform(cruce[["nbi_2022_pond"]])
    cruce["cpc_norm_inv"] = 1 - scaler.fit_transform(cruce[["credito_per_capita"]])
    cruce["indice_brecha"] = (
        (cruce["nbi_norm"] + cruce["cpc_norm_inv"]) / 2
    ).round(3)

    return cruce.sort_values("indice_brecha", ascending=False).reset_index(drop=True)

# ============================================================
# SIDEBAR: CARGA DE DATOS
# ============================================================
st.sidebar.header("📂 Datos")
st.sidebar.markdown("Sube un nuevo CSV del MAG para actualizar el ranking.")

archivo_mag = st.sidebar.file_uploader(
    "CSV del MAG (separador ;)",
    type=["csv"],
    help="Formato: mag_creditopublicoagropecuario_XXXXXX.csv"
)

# ID del archivo CSV del MAG en Google Drive
DRIVE_FILE_ID = "1pfX8zSG-uXM9s7kkhwoVf09sSJ9maGNe"

if archivo_mag is not None:
    # Si el usuario sube un archivo nuevo, lo usamos directamente
    df_mag_raw = pd.read_csv(archivo_mag, encoding="utf-8-sig", sep=";")
    st.sidebar.success(f"✅ Archivo cargado: {len(df_mag_raw):,} registros")
else:
    # Si no, descargamos el CSV por defecto desde Google Drive
    @st.cache_data
    def cargar_mag_drive(file_id):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url)
        return pd.read_csv(
            io.StringIO(r.content.decode("utf-8-sig")),
            sep=";"
        )
    df_mag_raw = cargar_mag_drive(DRIVE_FILE_ID)
    st.sidebar.info("📌 Usando dataset por defecto (MAG 2013–2025)")

# ============================================================
# PROCESAMIENTO
# ============================================================
with st.spinner("Procesando datos..."):
    df_nbi   = cargar_nbi()
    gdf_prov = cargar_geojson()
    # Si el usuario subió un archivo incluimos todos los años (incluyendo 2026)
    # Si usamos el dataset por defecto excluimos el año incompleto
    df_mag   = procesar_mag(df_mag_raw, incluir_anio_incompleto=archivo_mag is not None)
    ranking  = calcular_ranking(df_mag, df_nbi)

    # Unimos ranking con GeoDataFrame
    gdf = gdf_prov.merge(
        ranking, left_on="province", right_on="DPA_DESPRO", how="left"
    )
    # Corregimos rep_x de Guayas si es NaN
    idx_g = gdf[gdf["province"] == "Guayas"].index
    if len(idx_g) > 0 and pd.isna(gdf.loc[idx_g[0], "rep_x"]):
        gdf.loc[idx_g[0], "rep_x"] = -79.60
        gdf.loc[idx_g[0], "rep_y"] = -2.20

anios_disponibles = sorted(df_mag["CP_ANIO"].unique())
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 ¿Cómo interpretar el índice?")
st.sidebar.markdown(
    "El **índice de brecha** combina dos factores con igual peso:\n\n"
    "- 🔴 **NBI 2022**: proporción de población pobre por necesidades básicas insatisfechas\n"
    "- 💰 **Crédito per cápita**: USD de crédito público recibido por habitante (2013-2025)\n\n"
    "Un índice de **0.85** significa que la provincia combina **alta pobreza** y "
    "**bajo crédito per cápita** — es la más prioritaria para banca de desarrollo.\n\n"
    "Un índice de **0.29** significa que la provincia está **bien atendida** en "
    "relación a su nivel de pobreza."
)
st.sidebar.markdown(f"**Registros procesados:** {len(df_mag):,}")

# ============================================================
# LAYOUT: 3 PESTAÑAS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🗺️ Mapa de priorización",
    "📊 Ranking territorial",
    "👩‍🌾 Brecha de género"
])

# ============================================================
# PESTAÑA 1: MAPA
# ============================================================
# ============================================================
# PESTAÑA 1: MAPA
# ============================================================
with tab1:
    st.subheader("Índice de brecha de atención por provincia")
    st.markdown(
        "Provincias en **rojo oscuro** tienen alta pobreza (NBI) y bajo crédito "
        "per cápita — son las más prioritarias para banca de desarrollo."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 9),
                              gridspec_kw={"width_ratios": [1, 3.5]})
    ax_gal = axes[0]
    ax     = axes[1]

    gdf_gal  = gdf[gdf["province"] == "Galápagos"]
    gdf_cont = gdf[gdf["province"] != "Galápagos"]

    # Panel izquierdo: Galápagos
    gdf_gal.plot(column="indice_brecha", cmap="YlOrRd", linewidth=0.8,
                 edgecolor="white", ax=ax_gal, vmin=0.3, vmax=0.9,
                 missing_kwds={"color": "lightgrey"})
    ax_gal.set_axis_off()

    # Etiqueta centrada EN el mapa de Galápagos (no arriba)
    val_gal = gdf_gal["indice_brecha"].values[0] if len(gdf_gal) > 0 else 0
    ax_gal.text(
        0.5, 0.5,
        f"Galápagos\nÍndice: {val_gal:.2f}",
        transform=ax_gal.transAxes,
        ha="center", va="center",
        fontsize=9, fontweight="bold"
    )

    # Panel derecho: Ecuador continental
    gdf_cont.plot(column="indice_brecha", cmap="YlOrRd", linewidth=0.8,
                  edgecolor="white", legend=True, ax=ax,
                  legend_kwds={"label": "Índice de brecha\n(mayor = más prioritaria)",
                               "shrink": 0.6,
                               "format": "%.2f"},
                  vmin=0.3, vmax=0.9,
                  missing_kwds={"color": "lightgrey"})

    # Título de la barra de color en negrilla e igual al título principal
    ax.get_figure().axes[-1].set_ylabel(
        "Índice de brecha\n(mayor = más prioritaria)",
        fontsize=12, fontweight="bold"
    )

    # Etiquetas de provincias: mismo estilo que Galápagos (negrilla)
    for _, row in gdf_cont.iterrows():
        if pd.notna(row.get("indice_brecha")) and pd.notna(row.get("rep_x")):
            ax.annotate(
                f"{row['province']}\n({row['indice_brecha']:.2f})",
                xy=(row["rep_x"], row["rep_y"]),
                ha="center",
                fontsize=7,
                fontweight="bold",
                color="black"
            )

    ax.set_axis_off()
    # Título más grande y en negrilla
    ax.set_title(
        f"Crédito público agropecuario — Ecuador {min(anios_disponibles)}–{max(anios_disponibles)}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# ============================================================
# PESTAÑA 2: RANKING
# ============================================================
with tab2:
    st.subheader("Ranking de priorización territorial")

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Alta prioridad", f"{(ranking['indice_brecha'] >= 0.75).sum()} provincias")
    col2.metric("🟡 Prioridad media", f"{((ranking['indice_brecha'] >= 0.55) & (ranking['indice_brecha'] < 0.75)).sum()} provincias")
    col3.metric("🟢 Baja prioridad", f"{(ranking['indice_brecha'] < 0.55).sum()} provincias")

    # Tabla interactiva
    tabla = ranking[[
        "DPA_DESPRO", "nbi_2022_pond", "credito_per_capita",
        "monto_millones", "indice_brecha"
    ]].copy()
    tabla.index = range(1, len(tabla)+1)
    tabla.columns = [
        "Provincia", "NBI 2022", "Créd/hab (USD)",
        "Monto total (M)", "Índice brecha"
    ]

    # Coloreamos por prioridad
    def color_fila(row):
        idx = row.name
        if idx <= 6:
            return ["background-color: #ffcccc; color: black"] * len(row)
        elif idx <= 18:
            return ["background-color: #fff3cc; color: black"] * len(row)
        else:
            return ["background-color: #ccedcc; color: black"] * len(row)

    st.dataframe(
        tabla.style.apply(color_fila, axis=1).format({
            "NBI 2022": "{:.3f}",
            "Créd/hab (USD)": "${:,.0f}",
            "Monto total (M)": "${:.1f}M",
            "Índice brecha": "{:.3f}"
        }),
        use_container_width=True,
        height=600
    )

    # Botón de descarga del ranking
    csv_ranking = tabla.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Descargar ranking CSV",
        data=csv_ranking,
        file_name="ranking_priorizacion.csv",
        mime="text/csv"
    )

# ============================================================
# PESTAÑA 3: BRECHA DE GÉNERO
# ============================================================
with tab3:
    st.subheader("Evolución de la brecha de género en el acceso al crédito")

    # Calculamos brecha de género
    df_personas = df_mag[df_mag["CP_GENERO"].isin(["Femenino", "Masculino"])].copy()

    genero_anual = df_personas.groupby(["CP_ANIO", "CP_GENERO"]).agg(
        monto=("CP_VALOR_USD", "sum")
    ).reset_index()
    genero_anual["monto_M"] = genero_anual["monto"] / 1_000_000

    pivot = genero_anual.pivot_table(
        index="CP_ANIO", columns="CP_GENERO",
        values="monto_M", aggfunc="sum"
    ).reset_index()
    pivot["pct_femenino"] = (
        pivot["Femenino"] / (pivot["Femenino"] + pivot["Masculino"]) * 100
    ).round(1)

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(pivot["CP_ANIO"], pivot["pct_femenino"],
             color="#E76F51", marker="o", linewidth=2.5)
    ax2.axhline(y=50, color="gray", linestyle="--", linewidth=1.2,
                label="Paridad (50%)")
    ax2.fill_between(pivot["CP_ANIO"], pivot["pct_femenino"], 50,
                     alpha=0.15, color="#E76F51")
    ax2.set_ylim(25, 55)
    ax2.set_xlabel("Año")
    ax2.set_ylabel("% crédito recibido por mujeres")
    ax2.set_title("Evolución del acceso femenino al crédito (personas naturales)")
    ax2.legend()
    st.pyplot(fig2)
    plt.close()

    # Métrica resumen
    ultimo = pivot.iloc[-1]
    st.info(
        f"📈 **Hallazgo principal**: en {int(pivot.iloc[0]['CP_ANIO'])} las mujeres recibían el "
        f"{pivot.iloc[0]['pct_femenino']}% del crédito agropecuario. En {int(ultimo['CP_ANIO'])} "
        f"llegaron al {ultimo['pct_femenino']}% — una mejora de "
        f"{ultimo['pct_femenino'] - pivot.iloc[0]['pct_femenino']:.1f} puntos porcentuales en 12 años. "
        f"El crédito público agropecuario avanza hacia la paridad de género (50%).")

    st.metric(
        label=f"% crédito femenino en {int(ultimo['CP_ANIO'])}",
        value=f"{ultimo['pct_femenino']}%",
        delta=f"{(ultimo['pct_femenino'] - pivot.iloc[0]['pct_femenino']):.1f}pp vs {int(pivot.iloc[0]['CP_ANIO'])}")

st.markdown("---")
st.markdown(
    "**Fuentes**: MAG — Crédito público agropecuario (2026) | "
    "INEC — Censo de Población y Vivienda 2022 | "
    "Elaboración propia como parte del TFM — UCM Data Science & Business Analytics"
    "[📂 Código fuente](https://github.com/AndresMera98/tfm-credito-agropecuario-ecuador)"
)
