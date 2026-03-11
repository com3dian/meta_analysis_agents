"""
This file contains predefined, named metadata standards that can be used by the orchestrator.

Includes standards for specific usecase.

Example usage with objective:

```python
from src.standards import METADATA_STANDARDS

objective = f'''You are an expert agricultural data extraction specialist. Your task is to extract **measured crop yield information** and related agronomic/context variables from scientific research papers. You must reason carefully and follow a multi-step extraction process to ensure accurate and complete data.

**MULTI-STEP REASONING PROCESS**

### STEP 1: Anchor Identification
Scan the entire content (including paragraphs, tables, captions, footnotes, and supplements) to locate all sentences or table entries that mention **actual measured crop yield values** (e.g., "maize yield was 7.2 t/ha"). Ignore model performance metrics (e.g., RMSE, R²), yield gaps, and simulated/modelled outputs.

### STEP 2: Contextual Reasoning
For each identified yield anchor, analyze its surrounding context (paragraphs, section headers, tables, captions) to extract the following associated fields:
- **Crop Type**: Specific crop name (e.g., maize, wheat, rice, soybean, corn, etc.).
- **Yield Value**: Actual yield measurement **with units** (e.g., 8.5 t/ha, 3500 kg/ha, 42 bu/acre).
- **Location**: Study location (prefer coordinates if reported; otherwise country/province/state/city/research station/farm name).
- **Year**: Data collection year or growing season (e.g., 2018, 2019–2020; not publication year).
- **Treatment (Water or Nutrients)**: The experimental treatment (e.g., Irrigation amount/frequency or total nitrogen fertilizer application amount and frequency).
- **Tillage (Density or Cultivation)**: Density and Planting (e.g., Row/plant spacing, density, etc.)
- **Soil Properties**: Soil properties of the experimental sites (e.g., Texture, organic matter, pH, electrical conductivity, total nitrogen, available phosphorus/potassium, bulk density, field water holding capacity, available water capacity, and soil depth)
- **Climate**: Climate data (e.g., daily/decadal Tmax/Tmin/Tmean, precipitation, radiation/sunshine, wind speed, VPD, ET0, GDD, KDD, cumulative precipitation, critical windows)
- **UAV Data**: Raw spectral data or vegetation index (e.g., NDVI, EVI, GNDVI, NDRE, WDRVI, MCARI, red edge index, etc.)

If the fields are explicitly stated in the anchor or nearby context, extract them directly.

### STEP 3: Completeness, Evidence & Confidence
If any required field for a yield record is **missing**, try to retrieve it from specific related sections:
- For missing **location** or **year**, check **Materials and Methods / Site description**.
- For missing **crop type**, check **Abstract** and **Materials and Methods**.
- For missing **treatment (water/fertilizer)**, check **Field management / Experimental design / Plot / Block / Treatment / Table**.
- For missing **tillage (density & cultivation)**, check **Planting / Agronomic practices / Cultivation / Management**.
- For missing **soil properties**, check **Soil / Site description / Materials and Methods / Table**.
- For missing **climate**, check **Climate / Weather / Meteorology / Data sources**.
- For missing **UAV data**, check **UAV / UAS / Flight / Sensor / Data acquisition / Table / Figure captions**.

If you cannot find a field with **reasonable certainty**, discard the entire record. Only keep records where **all required fields are present and consistent**.

### STEP 4: Record Construction
For each complete and validated record, return a structured JSON entry following the schema below. Keep original units and expressions; **do not convert**. Put the ≤30-word evidence snippet for the **yield** in `"yield_notes"`. If an associated field is not found with reasonable certainty, set its value to `null` but still populate its source_section, confidence, and notes when applicable. Apply de-duplication across records where (crop_type, yield_value+unit, location, year, treatment if any) are identical.

**TARGET DATA TO EXTRACT**:
- ✅ INCLUDE: Actual field-measured or reported yields (e.g., from experiments, harvest trials)
- ❌ EXCLUDE: Model evaluation metrics (RMSE, R², MAE), yield gaps, predictions, correlation coefficients

**META-ANALYTIC SCHEMA**:
{METADATA_STANDARDS["climate_vs_cropyield"]}

**OUTPUT FORMAT**: Return results in JSON format with an array of yield records following the schema above. If no yield data is found, return an empty array. Return ONLY the JSON response, no additional text or explanations.'''

orchestrator.run(
    source="./data/papers",
    objective=objective,
    name="crop_yield_extraction"
)
```
"""

METADATA_STANDARDS = {
    "wopke_100": """
{
    "Year of data": "Year(s) when the experiment data were collected. Example: 2018 or 2017–2019.",
    "Duration of experiment": "Total time the experiment ran from first sowing to final harvest. Example: 120 days or 2 growing seasons.",
    "Experimental design": "Type of experimental layout used. Example: Randomized Complete Block Design (RCBD).",
    "Sowing date 1": "Date when Crop species 1 (the first species listed in the intercropping system) was sown. Example: Zea mays sown 15 April 2019.",
    "Sowing date 2": "Date when Crop species 2 (the second species listed in the intercropping system) was sown. Example: Glycine max sown 20 April 2019.",
    "Harvest date 1": "Date when Crop species 1 (the first species listed in the intercropping system) was harvested. Example: Zea mays harvested 10 September 2019.",
    "Harvest date 2": "Date when Crop species 2 (the second species listed in the intercropping system) was harvested. Example: Glycine max harvested 25 August 2019.",
    "Lat": "Latitude of the experimental site in decimal degrees. Example: 35.12.",
    "Lon": "Longitude of the experimental site in decimal degrees. Example: -1.56.",
    "Crop species 1": "Common or scientific name of the first crop species in the intercropping system (the first species mentioned in the pair). Example: Wheat or Triticum aestivum.",
    "Crop species 2": "Common or scientific name of the second crop species in the intercropping system (the second species mentioned in the pair). Example: Soybean or Glycine max.",
    "Crop type 1": "Agronomic functional type of Crop species 1 inferred from the species (e.g., cereal, legume, oilseed, root crop). Example: cereal.",
    "Crop type 2": "Agronomic functional type of Crop species 2 inferred from the species (e.g., cereal, legume, oilseed, root crop). Example: legume.",
    "Intercropping pattern": "Spatial arrangement of the two crops in the intercrop; classify as Row (alternating rows of each crop), Strip (blocks or strips with multiple rows per crop), or Mixed (species mixed within the same row or stand). Example: Row.",
    "Density ic 1": "Plant density of Crop species 1 in the intercropping treatment. Example: 5.",
    "Density ic 2": "Plant density of Crop species 2 in the intercropping treatment. Example: 20.",
    "Density sc 1": "Plant density of Crop species 1 when grown as a sole crop (monoculture). Example: 8。",
    "Density sc 2": "Plant density of Crop species 2 when grown as a sole crop (monoculture). Example: 30.",
    "N input SC1": "Nitrogen fertilizer applied to the sole crop treatment of Crop species 1. Example: 120.",
    "N input SC2": "Nitrogen fertilizer applied to the sole crop treatment of Crop species 2. Example: 30.",
    "N input IC1": "Nitrogen fertilizer attributed to Crop species 1 in the intercropping treatment. Example: 60.",
    "N input IC2": "Nitrogen fertilizer attributed to Crop species 2 in the intercropping treatment. Example: 30.",
    "N total in IC": "Total nitrogen fertilizer applied in the intercropping treatment. Example: 90.",
    "N Unit": "Unit used for nitrogen fertilizer amounts. Example: kg N ha−1.",
    "P input SC1": "Phosphorus fertilizer applied to the sole crop treatment of Crop species 1. Example: 40.",
    "P input SC2": "Phosphorus fertilizer applied to the sole crop treatment of Crop species 2. Example: 30.",
    "P input IC1": "Phosphorus fertilizer attributed to Crop species 1 in the intercropping treatment. Example: 20.",
    "P input IC2": "Phosphorus fertilizer attributed to Crop species 2 in the intercropping treatment. Example: 20.",
    "P total in IC": "Total phosphorus fertilizer applied in the intercropping treatment. Example: 40.",
    "P Unit": "Unit used for phosphorus fertilizer amounts. Example: kg P2O5 ha−1.",
    "K input SC1": "Potassium fertilizer applied to the sole crop treatment of Crop species 1. Example: 60.",
    "K input SC2": "Potassium fertilizer applied to the sole crop treatment of Crop species 2. Example: 50.",
    "K input IC1": "Potassium fertilizer attributed to Crop species 1 in the intercropping treatment. Example: 30.",
    "K input IC2": "Potassium fertilizer attributed to Crop species 2 in the intercropping treatment. Example: 30.",
    "K total in IC": "Total potassium fertilizer applied in the intercropping treatment. Example: 60.",
    "K Unit": "Unit used for potassium fertilizer amounts. Example: kg K2O ha−1.",
    "Data source": "Location of the extracted data within the publication. Example: Table 1,2; Figure 3; Supplementary Table S1.",
    "unified yield sc 1": "Yield of Crop species 1 grown as a sole crop, converted to a common yield unit. Example: 8.5.",
    "unified yield sc 2": "Yield of Crop species 2 grown as a sole crop, converted to a common yield unit. Example: 2.4.",
    "unified yield ic 1": "Yield of Crop species 1 when grown in the intercropping treatment. Example: 6.1.",
    "unified yield ic 2": "Yield of Crop species 2 when grown in the intercropping treatment. Example: 1.8.",
    "Yield unit": "Unit used for all standardized yields. Example: t ha−1."
}
""",
    "climate_vs_cropyield": """
{
    "crop_type": "specific crop name (e.g., maize, wheat, rice, soybean, corn)",
    "yield_value": "numerical value with unit (e.g., 8.5 t/ha, 3500 kg/ha, 42 bu/acre)",
    "location": "study location (prefer coordinates if reported; otherwise country/province/state/city/research station/farm name)",
    "year": "data collection year or growing season (e.g., 2018, 2019–2020; not publication year)",
    
    "yield_source_section": "section where yield was found (e.g., Materials and Methods, Results, Table 1)",
    "yield_confidence": "high | medium | low",
    "yield_notes": "≤30-word evidence snippet and any clarification",
    
    "Treatment": "Water or Nutrients of the experiment (e.g., irrigation type/amount/timing; N total/splits; N–P–K if reported)",
    "Treatment_source_section": "section where treatment was found",
    "Treatment_confidence": "high | medium | low",
    "Treatment_notes": "any additional context or clarification",
    
    "Tillage": "Density or Cultivation (e.g., row/plant spacing, density, tillage system, key dates)",
    "Tillage_source_section": "section where tillage info was found",
    "Tillage_confidence": "high | medium | low",
    "Tillage_notes": "any additional context or clarification",
    
    "Soil_property": "soil physico-chemical properties (e.g., texture, OM, pH, EC, N/P/K, bulk density, FC/AWC, depth)",
    "Soil_source_section": "section where soil properties were found",
    "Soil_confidence": "high | medium | low",
    "Soil_notes": "any additional context or clarification",
    
    "climate": "climate conditions (e.g., source/scale; Tmax/Tmin/Tmean, precipitation, radiation/sunshine, wind, VPD, ET0; GDD/KDD; key windows)",
    "climate_source_section": "section where climate data was found",
    "climate_confidence": "high | medium | low",
    "climate_notes": "any additional context or clarification",
    
    "remote_sensing_data": "UAV spectral data (e.g., platform/sensor/bands, AGL, GSD, flight dates, calibration/GCP; indices/values if any)",
    "rs_source_section": "section where remote sensing data was found",
    "rs_confidence": "high | medium | low",
    "rs_notes": "any additional context or clarification"
}
""",
    
    # Example output format template
    "climate_vs_cropyield_output_format": """
{
    "yield_records": [
        {
            "crop_type": "specific crop name",
            "yield_value": "numerical value with unit",
            "location": "study location",
            "year": "data collection year",
            "yield_source_section": "section where found",
            "yield_confidence": "high/medium/low",
            "yield_notes": "≤30-word evidence snippet",
            "Treatment": "Water or Nutrients description",
            "Treatment_source_section": "section where found",
            "Treatment_confidence": "high/medium/low",
            "Treatment_notes": "clarification",
            "Tillage": "Density or Cultivation description",
            "Tillage_source_section": "section where found",
            "Tillage_confidence": "high/medium/low",
            "Tillage_notes": "clarification",
            "Soil_property": "soil properties description",
            "Soil_source_section": "section where found",
            "Soil_confidence": "high/medium/low",
            "Soil_notes": "clarification",
            "climate": "climate conditions description",
            "climate_source_section": "section where found",
            "climate_confidence": "high/medium/low",
            "climate_notes": "clarification",
            "remote_sensing_data": "UAV spectral data description",
            "rs_source_section": "section where found",
            "rs_confidence": "high/medium/low",
            "rs_notes": "clarification"
        }
    ]
}
"""
}
