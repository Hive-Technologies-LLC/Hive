# ============================================================
# Hive Battery Analyzer — services/analyze.py
# Orchestration: normalize -> profile -> SOC/SOH -> faults -> result
# ============================================================

from .normalize import normalize_dataframe_from_upload
from .profile import profile_data
from .soc import estimate_soc_ocv, estimate_soc_coulomb, soc_confidence, blend_soc
from .soh import soh_from_cycles, soh_from_temperature_proxy, soh_from_resistance_proxy, soh_confidence
from .faults import flag_cells
from .stats import score_data_quality, basic_stats

def analyze_content(filename: str, content: bytes, analyzer_version: str):
    # Parse -> DataFrame
    df, debug = normalize_dataframe_from_upload(filename.lower(), content)

    if not df.empty:
        df = df.dropna(how="all")

    prof = profile_data(df)

    # SOC
    soc_ocv, info_ocv = estimate_soc_ocv(df)
    soc_cc,  info_cc  = estimate_soc_coulomb(df)

    conf_soc = soc_confidence(prof)
    soc_hybrid = blend_soc(soc_ocv, conf_soc["ocv"], soc_cc, conf_soc["cc"])

    soc_candidates = []
    if soc_ocv is not None: soc_candidates.append(("OCV", soc_ocv, conf_soc["ocv"], info_ocv))
    if soc_cc  is not None: soc_candidates.append(("Coulomb", soc_cc, conf_soc["cc"], info_cc))
    if soc_hybrid is not None:
        soc_candidates.append(("Hybrid", soc_hybrid, max(conf_soc["ocv"], conf_soc["cc"]), {"blend": "weighted"}))

    if soc_candidates:
        soc_method, soc_value, soc_conf_best, _ = max(soc_candidates, key=lambda x: x[2])
    else:
        soc_method, soc_value, soc_conf_best = (None, None, 0.0)

    # SOH
    soh_cyc, cyc_info = soh_from_cycles(df)
    soh_tmp, tmp_info = soh_from_temperature_proxy(df)
    soh_res, res_info = soh_from_resistance_proxy(df)

    conf_soh = soh_confidence(prof)

    soh_candidates = []
    if soh_cyc is not None: soh_candidates.append(("Cycles", soh_cyc, conf_soh["cycles"], cyc_info))
    if soh_tmp is not None: soh_candidates.append(("TempProxy", soh_tmp, conf_soh["temp"], tmp_info))
    if soh_res is not None: soh_candidates.append(("Resistance", soh_res, conf_soh["resistance"], res_info))

    if soh_candidates:
        soh_method, soh_value, soh_conf_best, _ = max(soh_candidates, key=lambda x: x[2])
    else:
        soh_method, soh_value, soh_conf_best = (None, None, 0.0)

    flags = flag_cells(df)
    tier = score_data_quality(df)
    stats = basic_stats(df)

    result = {
        "version": analyzer_version,
        "SOC (%)": round(soc_value, 2) if soc_value is not None else None,
        "SOH (%)": round(soh_value, 2) if soh_value is not None else None,
        "Flagged Cells": flags,
        "Accuracy Tier": tier,
        "methods": {
            "soc": {
                "method_used": soc_method,
                "confidence": round(float(soc_conf_best), 3),
                "candidates": {
                    "OCV": {"value": None if soc_ocv is None else round(soc_ocv, 2), "confidence": round(conf_soc["ocv"], 3), "info": info_ocv},
                    "Coulomb": {"value": None if soc_cc is None else round(soc_cc, 2), "confidence": round(conf_soc["cc"], 3), "info": info_cc},
                    "Hybrid": {"value": None if soc_hybrid is None else round(soc_hybrid, 2), "confidence": round(max(conf_soc["ocv"], conf_soc["cc"]), 3), "info": {"blend": "weighted"}}
                }
            },
            "soh": {
                "method_used": soh_method,
                "confidence": round(float(soh_conf_best), 3),
                "candidates": {
                    "Cycles": {"value": None if soh_cyc is None else round(soh_cyc, 2), "confidence": round(conf_soh["cycles"], 3), "info": cyc_info},
                    "TempProxy": {"value": None if soh_tmp is None else round(soh_tmp, 2), "confidence": round(conf_soh["temp"], 3), "info": tmp_info},
                    "Resistance": {"value": None if soh_res is None else round(soh_res, 2), "confidence": round(conf_soh["resistance"], 3), "info": res_info}
                }
            }
        },
        "stats": stats,
        "errors": [],
        "debug": {
            "profile": prof,
            "columns_after": debug.get("columns_after", []),
            "units_hint": debug.get("units_hint", {})
        }
    }

    return result, prof, tier, stats