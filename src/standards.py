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
