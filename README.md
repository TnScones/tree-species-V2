import json
import streamlit as st

from species_verifier.core import run_plausibility_check

st.set_page_config(page_title="Tree Species Plausibility Checker", layout="wide")
st.title("Tree species plausibility checker")

st.caption(
    "This app performs a *plausibility* check (not definitive species ID) by combining public biodiversity occurrences (GBIF) "
    "with public remote-sensing layers (ESA WorldCover + Sentinel-2 NDVI via Microsoft Planetary Computer)."
)

claimed_species = st.text_input("Claimed species (scientific name)", value="Quercus robur")

st.markdown("### Plot boundary (GeoJSON)")
boundary_text = st.text_area(
    "Paste a GeoJSON Feature or Polygon geometry (EPSG:4326 lon/lat)",
    height=220,
    value='{"type":"Polygon","coordinates":[[[0.0,51.5],[0.01,51.5],[0.01,51.51],[0.0,51.51],[0.0,51.5]]]}',
)

col1, col2, col3, col4 = st.columns(4)
skip_satellite = col1.checkbox("Skip satellite steps", value=False)
worldcover_year = col2.selectbox("WorldCover year", [2021, 2020], index=0)
nearby_radius_km = col3.slider("Nearby GBIF radius (km)", 5, 200, 50)
max_occ = col4.slider("GBIF max records (paged)", 300, 3000, 900, step=300)

with st.expander("Advanced settings"):
    st.write("**Tip:** Satellite steps may be slower and require outbound internet access.")
    st.write("You can cache results server-side in Streamlit Cloud by enabling Streamlit caching (already used in core).")

if st.button("Run check"):
    try:
        boundary_geojson = json.loads(boundary_text)
    except Exception as e:
        st.error(f"Could not parse GeoJSON: {e}")
        st.stop()

    try:
        with st.spinner("Running plausibility check…"):
            report = run_plausibility_check(
                claimed_species=claimed_species,
                boundary_geojson=boundary_geojson,
                skip_satellite=skip_satellite,
                worldcover_year=int(worldcover_year),
                nearby_radius_km=float(nearby_radius_km),
                gbif_max_records=int(max_occ),
            )

        verdict = report["plausibility"]["verdict"]
        score = report["plausibility"]["score_0_to_1"]

        st.success("Done")

        st.subheader("Verdict")
        st.metric("Plausibility score", f"{score:.3f}", verdict)

        st.subheader("Key evidence")
        c1, c2, c3 = st.columns(3)
        c1.metric("GBIF records inside polygon", report["gbif"]["occurrences_in_polygon_count"])
        c2.metric("GBIF records nearby (bbox)", report["gbif"]["occurrences_nearby_count"])
        if report.get("satellite", {}).get("sentinel2_ndvi"):
            c3.metric("Sentinel-2 NDVI median", f"{report['satellite']['sentinel2_ndvi']['ndvi_median']:.3f}")
        else:
            c3.metric("Sentinel-2 NDVI median", "—")

        if report.get("satellite", {}).get("errors"):
            st.warning("Satellite step warnings/errors")
            st.write(report["satellite"]["errors"])

        st.subheader("Full report (JSON)")
        st.json(report)

        st.download_button(
            "Download report as JSON",
            data=json.dumps(report, indent=2),
            file_name="species_plausibility_report.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(str(e))
