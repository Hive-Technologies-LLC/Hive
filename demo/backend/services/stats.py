# ============================================================
# Hive Battery Analyzer — services/stats.py
# Data quality tier + basic stats
# ============================================================

def score_data_quality(df):
    score = sum(1 for f in ["voltage", "temp", "current_now", "cycle_count"] if f in df.columns)
    if score >= 4: return "Tier 3 – Full"
    if score == 3: return "Tier 2 – Enhanced"
    if score == 2: return "Tier 1 – Basic"
    return "Tier 0 – Minimal"

def basic_stats(df):
    def safe_mean(col):
        if col not in df.columns or df[col].dropna().empty:
            return None
        return float(df[col].mean())

    return {
        "voltage_avg_v": safe_mean("voltage"),
        "current_avg_a": safe_mean("current_now"),
        "temp_avg_c": safe_mean("temp"),
    }