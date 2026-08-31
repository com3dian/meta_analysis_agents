"""
Static map from ``data/wopke_100/paper_output`` folder names → GT ``Study#``.

Ground truth: ``data/wopke_paper_code/Database for combined sample 2015-03-05.csv``.

Built with author + title (+ year) matching, including known filename typos
(Helenius/J.Helenius, Marthin/MARTIN, Mustsaers/MUTSAERS, Milyazawa/Miyazawa,
Hauggaard-Nielson/Nielsen, Sslal/Dua). Assignments are 1:1 (one folder per study).

90 / 100 folders map to a Study#. Unmapped Study# are 91–100 (no matching PDF
in this corpus). Unmapped folders are papers present on disk but not in that
GT study set — see :data:`UNMAPPED_FOLDERS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

# folder name (exact ``paper_output/<folder>/``) → Study#
FOLDER_TO_STUDY_ID: Dict[str, int] = {
    "Jensen 1996 Grain yield, symbiotic N2 fixation and interspecific competition for inorganic N in pea-barley": 1,
    "Hauggaard-Nielsen 2001 Interspecific competition, N use and interference with weeds in pea-barley": 2,
    "Bulson 1997 Effects of plant density on intercropped wheat and field beans in an organic farming system": 3,
    "Li 2001 Wheat-maize or wheat-soybean strip intercropping I. Yield advantage and inerspecific interactions": 4,
    "Li 1999 Interspecific complementary and competitive interactions between intercropped maize and faba bean": 5,
    "Hauggaard-Nielson 2003 The comparison of nitrogen use and leaching in sole cropped versus intercropped": 6,
    "Hauggaard-Nielsen 2001 Evaluating pea and barley cultivars for complementarity in intercropping": 7,
    "Li et al 2006 Root distribution and interactions between intercropped species": 8,
    "Ghosh 2004 Growth, yield, competition and economics of groundnut-cereal fodder intercropping systems": 9,
    "Dhima 2006 Competition indices of common vetch and cereal intercrops in two seeding ratio": 10,
    "Andersen 2004 Biomass production, symbiotic nitrogen fixation and inorganic N use": 11,
    "Baumann 2001 Competition and crop performance in a leek celery intercropping system": 12,
    "Banik 2006 Wheat and chickpea intercropping systems in an additive series experiment": 13,
    "Corre-Hellou 2006 Interspecific competition for soil N and its interaction with N-2 fixation": 14,
    "Chu 2004 Nitrogen fixation and N transfer from peanut to rice cultivated in aerobic soil in an intercropping": 15,
    "Fan et al 2006 Nitrogen fixation of faba bean interacting with a non-legume in two contrasting intercropping": 16,
    "Agegnehu 2006 Yield performance and land use efficiency of barley and faba bean mixed cropping": 17,
    "Hauggaard-Nielsen et al 2006 Density and relative frequency effects on  pea barley intercrops": 18,
    "Reddy et al 1981Growth and resource use studies in an intercrop of pearl millet-groundnut": 19,
    "Awal et al 2006 Radiation interception and use by maize-peanut intercrop canopy": 20,
    "Waterer et al 1994 Yield and symbiotic nitrogen fixation in a pea-mustard intercrop as influenced": 21,
    "Song et al 2007 Effect of intercropping on crop yield and chemical and microiological properties": 22,
    "Banik et al 2000 Evaluation of mustard and legume intercropping": 23,
    "Watiki et al 1993 Radiation interception and growth of maize-cowpea intercrop": 24,
    "Olasantan et al 1994 Effects of itnercropping and fertilizer application on weed control and performance": 25,
    "Zhang et al 2007 Growth, yield and quality of wheat and cotton in relay strop intercropping systems": 26,
    "Haymes et al 1999 Competition between autumn and spring planted grain inetrcrops of wheat and field bean": 27,
    "Ghaley et al 2005 Intercropping of wheat and pea as influenced by nitrogen fertilization": 28,
    "Tobita et al 1994 Field evaluation of nitrogen fixation and use of nitrogen fertilizer": 29,
    "Carruthers et al 2000 Intercropping corn with soybean, lupin and forages": 30,
    "Lithourgidis et al 2007 Sustainable production of barley and wheat by intercropping common vetch": 31,
    "Knudsen et al 2004 Comparison of interspecific competition and N use in intercrops grown a": 32,
    "Chabi-Olaye et al 2005 Relationships of intercropped maize, stem borer damage to maize yield and land-use": 33,
    "Helenius et al 1994 Yield advantage and competition in intercropped oats and faba bean": 34,
    "Carr et al 1995 Grain yield and weed biomass of a wheat-lentil intercrop": 35,
    "Marthin et al 1990 Intercropping corn and soybean for silage in a cool temperate region": 36,
    "Bedoussac et al 2010 The efficiency of a durum wheat-winter pea intercrop": 37,
    "Jahansooz et al 2006 Radiation and water use associated with growth and yields of wheat and chickpea": 38,
    "Gunes et al 2007 Mineral nutrition of wheat, chickpea and lentil as affected by mixed cropping and soil mo": 39,
    "Ofori et al 1988 Maize-cowpea intercrop system, effect of nitrogen fertilizer on productivity and efficiency": 40,
    "Willey et al 1981 A field technique for separating above and below ground interactions in intercropping": 41,
    "Ntare 1990 Intercropping morphologically different cowpeas with pearl-,illite": 42,
    "Chowdhury et al 1994 Comparison of nitrogen, phosphorus and potassium utilization": 43,
    "Schmidtke et al Soil and atmospheric nitrogen uptake by lentil and barley as monocrops and intercrops": 44,
    "Akanvou et al 2001 Evaluating the use of two contrasting legume species as realy intercrop": 45,
    "Moynihan et al 1996 Intercropping annual medic with conventional height and semidwarf barley grown for grain": 46,
    "Reynolds et al 1994 Intercropping wheat and barley with N fixing legume species": 47,
    "Ofori et al 1987 Evaluation of N fixation and nitrogen economy of a maize-cowpea intercrop system using N": 48,
    "Ofori et al 1987 Relative sowing time and density of component crops in a maize cowpea intercrop system": 49,
    "Ofori et al 1987 The combined effects of nitrogen fertilizer and density of the legume component on": 50,
    "1. Mason 1986 Cassava-cowpea and cassava-peanut intercropping II Leaf afrea index and dry matter accumulation": 51,
    "3. Ong et al 1991The microclimate and productivity of a groudnut-millet intercrop during the rainy searson": 52,
    "4. Vyas 2006 Productivity and economics of integrated nutrient management in soybean (Glycine max) plus pigeonpea": 53,
    "5. Morgado 2008 Optimum plant population for maize-bean intercropping system in the Brazilian semi-arid region": 54,
    "8. Reddy et al 1990 Genotype effects in millet cowpea intercropping": 55,
    "11. Behera et al 2002 Biological and economical feasibility of intercroppign vegetables": 56,
    "12. Giri 1990 Studies on pigeonpea and groundnut intecropping under rainfed conditions": 57,
    "13. Ossom et al 2005 intercropping maize with grain legumes influences weed suppression": 58,
    "14. Tomar et al 1987 Effect of planting patterns in pigeonpea and soybean intercropping system": 59,
    "15. Ahmed et al 2000 Studies on yield, land equivalent ratio and crop performance rate in maize-mungbean intercropping": 60,
    "16. Sarkar et al 2000 Production potential and economic feasibility and sasame based intercropping system": 61,
    "17. Subramanian et al 1987 Intercropping effects on yield components of dryland sorghum, pigeon pea and mung bean": 62,
    "18. Mustsaers 1978 Mixed cropping experiements with maize and groudnuts": 63,
    "19. Ogbuehi et al 1987 Intercropping carrot and sweetcorn in a multiple cropping system": 64,
    "20. Mei et al 2012 Maize-faba bean intercropping with rhizobia inoculation enhances productivity and recovery": 65,
    "21. Ciftci et al 2005 Economic benefits ofmixed cropping of lentil with wheat and barley": 66,
    "23. Morgado et al 2003 Effects of plant population and nitrogen fertilizer on yield and efficiency": 67,
    "24. Subedi 1998 Profitability of barley and peas mixed intercropping in the subsistence farming systems": 68,
    "26. Gao et al 2010 Distribution of roots and root length density": 69,
    "27. Das et al 1991 Studies on pigeonpea and groundnut intercropping system": 70,
    "29. Nelson et al 2012 Yield and weed suppression of crop mixtures in organic": 71,
    "30. Agegnehu et al 2008 Yield potential and land use efficiency of wheat and faba bean mixed intercropping": 72,
    "31. Mason et al 1987 Intercropping in a temperate environment for irrigated fodder production": 73,
    "32. Lithourgidis et al 2011 Dry matter yield, nitrogen content and competition in pea-cereal intercropping systems": 74,
    "33. Prasad et al 1991 Pigeonpea and soybean intercropping systems under rainfed situation": 75,
    "34. Asl et al 2009 Potato and pinto bean intercropping based on replacement method": 76,
    "36. Kontturi et al 2011 Pea-oat intercrops to sustain lodging resistance and yield formation": 77,
    "37. Teasdale et al 1987 Performance of four tomato cultivars intercropped with snap beans": 78,
    "38. Mondal et al 2004 Effect of K on soil fertility and productivity under intercropped soya bean and sesame": 79,
    "39. Rees 1986 Crop growth, development and yield in semi-arid conditions in botswana": 80,
    "40. Gao et al 2009 Crop coefficiennt and water use efficiency of winter wheat-spring maize strip intercropping": 81,
    "41. Allen et al 1983 Yield of corn, cowpea, and soybean under different intercropping systems": 82,
    "42. Lei et al 2005 Water use efficiency of a mixed cropping system of corn with grasses": 83,
    "44. Milyazawa et al 2010 Intercropping green manure crops-effects on rooting patterns": 84,
    "45. Chang et al 1985 An analysis of competition between intercropped cowpea and maize": 85,
    "46. Neumann et al 2009 Evaluation of yield-density relations and optimization of intercrop compositions": 86,
    "47. Silwana et al 2007 The effects of inorganic and organic fertilizers on the growth and development of component crops": 87,
    "48. Aggarwal et al 1992 Resource use and plant interactions in a rice-mungbean intercrop": 88,
    "49. Sslal et al 2005 Production potential and competition indices in potato and french bean intercropping system": 89,
    "50. Kumar et al 2003 Biological and economical sustainability of forage": 90,
}

STUDY_ID_TO_FOLDER: Dict[int, str] = {sid: folder for folder, sid in FOLDER_TO_STUDY_ID.items()}

UNMAPPED_STUDY_IDS: List[int] = list(range(91, 101))

UNMAPPED_FOLDERS: List[str] = [
    "10. Pal et al 2000 Weed control studiesin pearlmillet",
    "2. Rezaei-Chianeh et al 2011 Intercropping of maize and faba bean at different plant population densities",
    "22. Chowdhury et al 1992 Utilization efficiency of applied nitrogen as related to yield advantage",
    "25. Ojikpong et al 2009 Effect of time of introducing sesame and nitrogen",
    "28. Robert-Nkrumah et al 1995 Performance of sweet potato cultivars intercropped with maize",
    "35. Balde et al 2011 Agronomic performance of no-tillage relay intercropping with maize under smallholder conditions",
    "43. Adhikary et al 1991 Studies on maize-legume intercropping and their residual effects on soil fertilizer",
    "6. Yilmaz 2008 Identification of advantages of maize-legume intercropping over solitary cropping",
    "7. Searle et al 1981 Effect of maize-legume intercropping systems and fertilizer nitrogen",
    "9. Ramakrishna et al 1994 Productivity and light interception upland rice-legume intercrops",
]


def study_id_for_folder(folder_name: str) -> Optional[int]:
    """Return GT ``Study#`` for a ``paper_output`` folder name, or ``None``."""
    return FOLDER_TO_STUDY_ID.get(folder_name)


def folder_for_study_id(study_id: int) -> Optional[str]:
    """Return ``paper_output`` folder name for a GT ``Study#``, or ``None``."""
    return STUDY_ID_TO_FOLDER.get(int(study_id))


def paper_folder_from_path(path: Union[str, Path]) -> str:
    """
    Resolve the paper folder name from a markdown/PDF path under ``paper_output``.

    Accepts either ``.../paper_output/<folder>/hybrid_auto/<file>.md`` or
    ``.../paper_output/<folder>``.
    """
    p = Path(path).resolve()
    parts = p.parts
    if "paper_output" in parts:
        idx = parts.index("paper_output")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # markdown under hybrid_auto: parent.parent is the paper folder
    if p.parent.name == "hybrid_auto":
        return p.parent.parent.name
    if p.is_file():
        return p.parent.name
    return p.name


def study_id_for_path(path: Union[str, Path]) -> Optional[int]:
    """Return GT ``Study#`` for a path under ``paper_output``, or ``None``."""
    return study_id_for_folder(paper_folder_from_path(path))


__all__ = [
    "FOLDER_TO_STUDY_ID",
    "STUDY_ID_TO_FOLDER",
    "UNMAPPED_STUDY_IDS",
    "UNMAPPED_FOLDERS",
    "study_id_for_folder",
    "folder_for_study_id",
    "paper_folder_from_path",
    "study_id_for_path",
]
