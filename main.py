"""
GeoNest — Bali Property Intelligence
Self-hosted FastAPI backend (replaces n8n + Claude API)
AI: Groq (free) with Llama 3.3 70B
"""

import os, json, math, csv, io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

# ── PATHS ──────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "data"
STATIC = BASE / "static"

# ── GROQ CLIENT ────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GROQ_MODEL  = "llama-3.3-70b-versatile"
ACCESS_CODE = os.environ.get("ACCESS_CODE", "B4liGo")

# ── IN-MEMORY DATA (loaded at startup) ─────────────────────
_data: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════

def load_geojson(name: str) -> dict:
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def load_population() -> dict[str, int]:
    pop = {}
    with open(DATA / "bali_population.tsv", encoding="utf-8") as f:
        lines = f.read().split("\n")[1:]  # skip header
    for line in lines:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) >= 3:
            subdistrict = cols[1].strip()
            val = cols[2].replace(",", "").replace(".", "").strip()
            try:
                pop[subdistrict] = int(val)
            except ValueError:
                pass
    return pop


def load_properties() -> list[dict]:
    props = []
    with open(DATA / "bali_properties_rent.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 6:
                continue
            try:
                lat = float(row[4].strip())
                lon = float(row[5].strip())
            except ValueError:
                continue
            props.append({
                "id":        int(row[0].strip()),
                "name":      row[1].strip(),
                "area":      row[2].strip(),
                "price_idr": int(row[3].strip()) if row[3].strip() else 0,
                "lat":       lat,
                "lon":       lon,
                "type":      "rent",
            })
    return props


# ══════════════════════════════════════════════════════════
# SECTION 2 — SPATIAL HELPERS
# ══════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_in_ring(lat: float, lon: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_of_feature(feature: dict, buffer: float = 0.05):
    geom = feature["geometry"]
    coords = []
    if geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                coords.extend(ring)
    elif geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            coords.extend(ring)
    elif geom["type"] == "MultiLineString":
        for line in geom["coordinates"]:
            coords.extend(line)
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lats) - buffer, max(lats) + buffer,
            min(lons) - buffer, max(lons) + buffer)


def point_in_feature(lat: float, lon: float, feature: dict) -> bool:
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        rings = geom["coordinates"]
    else:  # MultiPolygon
        rings = [ring for poly in geom["coordinates"] for ring in poly]
    for ring in rings:
        if point_in_ring(lat, lon, ring):
            return True
    return False


# ══════════════════════════════════════════════════════════
# SECTION 3 — RISK SPATIAL INDEX
# ══════════════════════════════════════════════════════════

def build_risk_index(geojson: dict) -> list:
    """Each entry: (min_lat, max_lat, min_lon, max_lon, label, ring, c_lat, c_lon)"""
    index = []
    for feature in geojson["features"]:
        rc = feature["properties"].get("Risk_Class", "")
        label = "low" if rc in ("Very Low", "Low") else ("medium" if rc == "Moderate" else "high")
        geom = feature["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            if not ring or len(ring) < 3:
                continue
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            c_lat = sum(lats) / len(lats)
            c_lon = sum(lons) / len(lons)
            index.append((min_lat, max_lat, min_lon, max_lon, label, ring, c_lat, c_lon))
    return index


def get_risk(lat: float, lon: float, index: list) -> tuple[str, str]:
    """Returns (label, source). source = 'exact' | 'estimated' | 'unclassified'"""
    for min_lat, max_lat, min_lon, max_lon, label, ring, _, _ in index:
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue
        if point_in_ring(lat, lon, ring):
            return label, "exact"
    min_dist, nearest_label = 999.0, "low"
    for min_lat, max_lat, min_lon, max_lon, label, ring, c_lat, c_lon in index:
        dist = haversine(lat, lon, c_lat, c_lon)
        if dist < min_dist:
            min_dist = dist
            nearest_label = label
    if min_dist <= 2.0:
        return nearest_label, "estimated"
    return "low", "unclassified"


# ══════════════════════════════════════════════════════════
# SECTION 4 — GEO LOOKUPS
# ══════════════════════════════════════════════════════════

def get_admin_info(lat: float, lon: float, admin_geojson: dict) -> dict:
    # bbox-filtered point-in-polygon
    for feature in admin_geojson["features"]:
        bbox = bbox_of_feature(feature, 0.05)
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
        if point_in_feature(lat, lon, feature):
            p = feature["properties"]
            return {
                "village":     p.get("Village", ""),
                "subdistrict": p.get("Subdistric", ""),
                "city":        p.get("City_Regen", ""),
                "province":    p.get("Province", "Bali"),
            }
    # fallback: nearest centroid
    min_dist, nearest = 999.0, None
    for feature in admin_geojson["features"]:
        geom = feature["geometry"]
        if geom["type"] == "MultiPolygon":
            coords = [c for poly in geom["coordinates"] for ring in poly for c in ring]
        else:
            coords = [c for ring in geom["coordinates"] for c in ring]
        if not coords:
            continue
        c_lon = sum(c[0] for c in coords) / len(coords)
        c_lat = sum(c[1] for c in coords) / len(coords)
        dist = haversine(lat, lon, c_lat, c_lon)
        if dist < min_dist:
            min_dist = dist
            nearest = feature
    if nearest:
        p = nearest["properties"]
        return {
            "village":     p.get("Village", ""),
            "subdistrict": p.get("Subdistric", ""),
            "city":        p.get("City_Regen", ""),
            "province":    p.get("Province", "Bali"),
        }
    return {"village": "", "subdistrict": "", "city": "", "province": "Bali"}


def get_nearest_road_distance(lat: float, lon: float, roads_geojson: dict) -> float:
    min_dist = 999.0
    bbox = 0.05
    for feature in roads_geojson["features"]:
        geom = feature["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        first = lines[0][0] if lines and lines[0] else None
        if first and abs(first[1] - lat) > bbox:
            continue
        for line in lines:
            for coord in line:
                d = haversine(lat, lon, coord[1], coord[0])
                if d < min_dist:
                    min_dist = d
                if min_dist < 0.05:
                    return round(min_dist, 2)
    return round(min_dist, 2)


def get_population_density(subdistrict: str, pop_data: dict[str, int]) -> dict:
    pop = pop_data.get(subdistrict)
    if not pop:
        sub_lower = subdistrict.lower()
        for k, v in pop_data.items():
            if k.lower() in sub_lower or sub_lower in k.lower():
                pop = v
                break
    if not pop:
        pop = 60000
    if pop < 60000:
        label = "low"
    elif pop < 120000:
        label = "medium"
    else:
        label = "high"
    return {"label": label, "value": pop}


def get_terrain_class(lat: float, lon: float, elevation_geojson: dict) -> dict:
    for feature in elevation_geojson["features"]:
        bbox = bbox_of_feature(feature)
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
        if point_in_feature(lat, lon, feature):
            p = feature["properties"]
            return {
                "value": p.get("value") or p.get("Value") or 1,
                "label": p.get("class_name", "Plains"),
            }
    return {"value": 1, "label": "Plains"}


def get_coastal_proximity(lat: float, lon: float, coastal_geojson: dict) -> dict:
    for feature in coastal_geojson["features"]:
        bbox = bbox_of_feature(feature, 0.5)
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
        if point_in_feature(lat, lon, feature):
            return {"label": "beachside", "distance_km": 0.0}
    min_dist = 999.0
    for feature in coastal_geojson["features"]:
        geom = feature["geometry"]
        if geom["type"] == "MultiPolygon":
            coords = [c for poly in geom["coordinates"] for ring in poly for c in ring]
        else:
            coords = [c for ring in geom["coordinates"] for c in ring]
        for coord in coords:
            d = haversine(lat, lon, coord[1], coord[0])
            if d < min_dist:
                min_dist = d
    dist_km = round(min_dist, 2)
    if dist_km <= 1:
        label = "beachside"
    elif dist_km <= 5:
        label = "near_beach"
    elif dist_km <= 15:
        label = "coastal_area"
    else:
        label = "inland"
    return {"label": label, "distance_km": dist_km}


def get_nearest_facility_distance(lat: float, lon: float, facilities: list[dict]) -> float:
    if not facilities:
        return 999.0
    min_dist = 999.0
    for fac in facilities:
        d = haversine(lat, lon, fac["lat"], fac["lon"])
        if d < min_dist:
            min_dist = d
        if min_dist < 0.1:
            break
    return round(min_dist, 2)


def get_nearest_any_facility(lat: float, lon: float, all_facilities: list[dict]) -> dict:
    min_dist, name, ftype = 999.0, "", ""
    for fac in all_facilities:
        d = haversine(lat, lon, fac["lat"], fac["lon"])
        if d < min_dist:
            min_dist = d
            name = fac.get("name", "")
            ftype = fac.get("type", "")
        if min_dist < 0.1:
            break
    return {"distance_km": round(min_dist, 2), "name": name, "type": ftype}


# ══════════════════════════════════════════════════════════
# SECTION 5 — RISK WARNING
# ══════════════════════════════════════════════════════════

def build_risk_warning(flood_risk: str, flood_src: str,
                        land_risk: str, land_src: str) -> dict:
    def emoji(r):
        return "🚨" if r == "high" else "⚠️" if r == "medium" else "✅"
    def src_note(s):
        return "" if s == "exact" else " (est.)" if s == "estimated" else " (n/a)"
    def flood_label(r, s):
        if r == "high":   return f"High flood risk — not recommended{src_note(s)}"
        if r == "medium": return f"Medium flood risk — proceed with caution{src_note(s)}"
        return f"Low flood risk — safe{src_note(s)}"
    def land_label(r, s):
        if r == "high":   return f"High landslide risk — not recommended{src_note(s)}"
        if r == "medium": return f"Medium landslide risk — proceed with caution{src_note(s)}"
        return f"Low landslide risk — safe{src_note(s)}"
    has_high   = flood_risk == "high"   or land_risk == "high"
    has_medium = flood_risk == "medium" or land_risk == "medium"
    risk_level = "danger" if has_high else "caution" if has_medium else "safe"
    risk_emoji = "🚨" if has_high else "⚠️" if has_medium else "✅"
    return {
        "flood":     {"level": flood_risk, "emoji": emoji(flood_risk),
                      "label": flood_label(flood_risk, flood_src), "source": flood_src},
        "landslide": {"level": land_risk,  "emoji": emoji(land_risk),
                      "label": land_label(land_risk, land_src), "source": land_src},
        "overall":   {"level": risk_level, "emoji": risk_emoji},
    }


# ══════════════════════════════════════════════════════════
# SECTION 6 — PROPERTY ENRICHMENT (done once at startup)
# ══════════════════════════════════════════════════════════

FACILITY_TYPES = ["hospital", "clinic", "school", "convenience",
                  "supermarket", "marketplace", "restaurant"]


def enrich_properties(properties: list[dict], geo: dict) -> list[dict]:
    admin_gj    = geo["admin"]
    roads_gj    = geo["roads"]
    flood_idx   = geo["flood_index"]
    land_idx    = geo["landslide_index"]
    elev_gj     = geo["elevation"]
    coastal_gj  = geo["coastal"]
    pop_data    = geo["population"]
    fac_by_type = geo["facilities_by_type"]
    all_fac     = geo["all_facilities"]

    enriched = []
    total = len(properties)
    for i, p in enumerate(properties):
        print(f"  Enriching property {i+1}/{total}: {p['name']}", flush=True)
        lat, lon = p["lat"], p["lon"]
        admin       = get_admin_info(lat, lon, admin_gj)
        flood_risk, flood_src = get_risk(lat, lon, flood_idx)
        land_risk,  land_src  = get_risk(lat, lon, land_idx)
        road_dist   = get_nearest_road_distance(lat, lon, roads_gj)
        pop         = get_population_density(admin["subdistrict"] or p["area"], pop_data)
        terrain     = get_terrain_class(lat, lon, elev_gj)
        coastal     = get_coastal_proximity(lat, lon, coastal_gj)
        nearest_any = get_nearest_any_facility(lat, lon, all_fac)

        fac_distances = {}
        for ftype in FACILITY_TYPES:
            fac_distances[ftype] = get_nearest_facility_distance(lat, lon, fac_by_type.get(ftype, []))

        enriched.append({
            **p,
            "subdistrict":           admin["subdistrict"] or p["area"],
            "city":                  admin["city"] or p["area"],
            "road_distance_km":      road_dist,
            "population_density":    pop["label"],
            "population_count":      pop["value"],
            "terrain_class":         terrain["value"],
            "terrain_label":         terrain["label"],
            "coastal_label":         coastal["label"],
            "beach_distance_km":     coastal["distance_km"],
            "facility_distances":    fac_distances,
            "nearest_any_facility":  nearest_any,
            "flood_risk":            flood_risk,
            "flood_risk_source":     flood_src,
            "landslide_risk":        land_risk,
            "landslide_risk_source": land_src,
            "risk_warning":          build_risk_warning(flood_risk, flood_src, land_risk, land_src),
        })
    return enriched


# ══════════════════════════════════════════════════════════
# SECTION 7 — SCORING ENGINE
# ══════════════════════════════════════════════════════════

def score_road(d: float) -> int:
    if d <= 1:  return 25
    if d <= 3:  return 15
    return 5

def score_facility(d: float) -> int:
    if d <= 0.5: return 25
    if d <= 1.5: return 20
    if d <= 3.0: return 13
    if d <= 5.0: return 7
    return 2

def score_density(label: str) -> int:
    return 20 if label == "low" else 12 if label == "medium" else 5

def score_terrain(value: int) -> int:
    return {3: 15, 4: 12, 2: 10, 5: 8}.get(value, 5)

def score_beach(label: str) -> int:
    return {"beachside": 15, "near_beach": 10, "coastal_area": 5}.get(label, 0)


def score_property(p: dict, filters: dict) -> dict:
    beach_requested = filters.get("beach_preference") == "close"
    requested_facs  = filters.get("nearby_facilities") or []

    road_pts = score_road(p["road_distance_km"])
    pop_pts  = score_density(p["population_density"])
    ter_pts  = score_terrain(p["terrain_class"])

    fac_pts = 0
    fac_detail = {}
    if requested_facs:
        total = 0
        for ftype in requested_facs:
            d   = p["facility_distances"].get(ftype, 999)
            pts = score_facility(d)
            total += pts
            fac_detail[ftype] = {"distance_km": d, "points": pts}
        fac_pts = round(total / len(requested_facs))
    else:
        nearest = p["nearest_any_facility"]
        fac_pts = score_facility(nearest["distance_km"])
        fac_detail["nearest"] = {
            "distance_km": nearest["distance_km"],
            "name": nearest["name"],
            "type": nearest["type"],
            "points": fac_pts,
        }

    beach_pts = score_beach(p["coastal_label"]) if beach_requested else 0

    raw       = road_pts + fac_pts + pop_pts + ter_pts + beach_pts
    max_pts   = 100 if beach_requested else 85
    final     = min(100, max(0, round((raw / max_pts) * 100)))

    return {
        "total": final,
        "breakdown": {
            "road":       {"points": road_pts, "distance_km": p["road_distance_km"]},
            "facility":   {"points": fac_pts,  "detail": fac_detail},
            "population": {"points": pop_pts,  "density": p["population_density"]},
            "terrain":    {"points": ter_pts,  "class": p["terrain_label"]},
            "beach":      {"points": beach_pts, "label": p["coastal_label"]} if beach_requested else {},
            "raw_score":    raw,
            "max_possible": max_pts,
        },
    }


# ══════════════════════════════════════════════════════════
# SECTION 8 — FILTER & RANK
# ══════════════════════════════════════════════════════════

ISLAND_AREAS = ["nusa penida", "nusa lembongan", "nusa ceningan"]

BALI_DIRECTIONAL_BOUNDS = {
    "east_bali":    (-8.80, -8.05, 115.45, 115.75),
    "west_bali":    (-8.70, -8.00, 114.40, 115.00),
    "north_bali":   (-8.25, -8.00, 114.50, 115.70),
    "south_bali":   (-8.85, -8.55, 115.00, 115.50),
    "central_bali": (-8.60, -8.25, 115.00, 115.40),
}

def resolve_directional(loc: str | None) -> str | None:
    if not loc:
        return None
    l = loc.lower().replace("-", "_").replace(" ", "_")
    if l in BALI_DIRECTIONAL_BOUNDS:
        return l
    if "east" in l or "timur" in l:    return "east_bali"
    if "west" in l or "barat" in l:    return "west_bali"
    if "north" in l or "utara" in l:   return "north_bali"
    if "south" in l or "selatan" in l: return "south_bali"
    if "centr" in l or "tengah" in l:  return "central_bali"
    return None


def filter_properties(enriched: list[dict], filters: dict) -> list[dict]:
    dir_key  = resolve_directional(filters.get("location"))
    dir_bbox = BALI_DIRECTIONAL_BOUNDS.get(dir_key) if dir_key else None

    out = []
    for p in enriched:
        # Budget
        if filters.get("max_budget_idr") and p["price_idr"] > filters["max_budget_idr"]:
            continue
        # Nusa Penida exclusion
        if filters.get("exclude_nusa_penida") is not False:
            is_island = any(i in p["area"].lower() or i in p["subdistrict"].lower()
                            for i in ISLAND_AREAS)
            if is_island:
                continue
        # Location
        loc = filters.get("location")
        if not loc or loc.lower() == "bali":
            pass
        elif dir_bbox:
            min_lat, max_lat, min_lon, max_lon = dir_bbox
            if not (min_lat <= p["lat"] <= max_lat and min_lon <= p["lon"] <= max_lon):
                continue
        else:
            loc_l = loc.lower()
            if not (loc_l in p["area"].lower() or
                    loc_l in p["city"].lower() or
                    loc_l in p["subdistrict"].lower() or
                    p["city"].lower() in loc_l):
                continue
        # Flood risk
        fr = filters.get("flood_risk", "any")
        if fr == "low"          and p["flood_risk"] != "low":    continue
        if fr == "medium"       and p["flood_risk"] == "high":   continue
        if fr == "medium_exact" and p["flood_risk"] != "medium": continue
        if fr == "high_exact"   and p["flood_risk"] != "high":   continue
        # Landslide risk
        lr = filters.get("landslide_risk", "any")
        if lr == "low"          and p["landslide_risk"] != "low":    continue
        if lr == "medium"       and p["landslide_risk"] == "high":   continue
        if lr == "medium_exact" and p["landslide_risk"] != "medium": continue
        if lr == "high_exact"   and p["landslide_risk"] != "high":   continue
        # Population density
        pd_ = filters.get("population_density", "any")
        if pd_ == "low"    and p["population_density"] != "low":  continue
        if pd_ == "medium" and p["population_density"] == "high": continue
        # Road distance
        max_road = filters.get("max_road_distance_km")
        if max_road and p["road_distance_km"] > max_road + 0.5:
            continue
        # Elevation
        elev = filters.get("elevation_preference", "any")
        if elev == "high" and p["terrain_class"] < 3: continue
        if elev == "low"  and p["terrain_class"] > 2: continue
        # Beach
        if filters.get("beach_preference") == "close":
            if p["beach_distance_km"] > 8 or p["coastal_label"] == "inland":
                continue
        # Facilities
        facs = filters.get("nearby_facilities") or []
        if facs:
            max_fac = filters.get("max_facility_distance_km") or 5
            if any(p["facility_distances"].get(ft, 999) > max_fac for ft in facs):
                continue
        out.append(p)
    return out


# ══════════════════════════════════════════════════════════
# SECTION 9 — LLM PROMPTS
# ══════════════════════════════════════════════════════════

FILTER_SYSTEM_PROMPT = """You are a geospatial property search assistant for Indonesia.
When a user describes what they want in a house, extract their criteria
and return ONLY a JSON object with these exact fields:

{
  "location": "specific area name or null if user says Bali generally",
  "max_budget_idr": number or null,
  "max_road_distance_km": number or null,
  "flood_risk": "low" or "medium" or "any",
  "landslide_risk": "low" or "medium" or "any",
  "population_density": "low" or "medium" or "any",
  "elevation_preference": "high" or "low" or "any",
  "property_type": "rent" or "buy" or "any",
  "beach_preference": "close" or "any",
  "nearby_facilities": [],
  "max_facility_distance_km": number or null,
  "exclude_nusa_penida": true or false,
  "is_exploratory": true or false
}

Rules:
- Reply ONLY with the JSON object, nothing else
- No markdown, no explanation, no code fences
- If a field is not mentioned, use null for numbers and "any" for strings
- Convert budget mentions like "5 juta" to 5000000, "500 juta" to 500000000
- If user mentions "cold", "cool", "sejuk", "dingin", or "highland", set location to null and add "elevation_preference": "high"
- If user mentions "Bali" as general location without specific area, set location to null
- If user mentions "beach", "pantai", "coastal", "laut", "seaside", "tepi pantai", set beach_preference to "close"
- Otherwise set beach_preference to "any"
- Extract facility mentions and map to these exact types:
  hospital → "hospital"
  clinic, puskesmas → "clinic"
  school → "school"
  indomaret, alfamart, minimarket, convenience store → "convenience"
  supermarket, hypermarket, carrefour → "supermarket"
  pasar, traditional market, market → "marketplace"
  restaurant, cafe, warung, makan → "restaurant"
- If no facilities mentioned, set nearby_facilities to []
- If max distance mentioned (e.g. "within 2km"), set max_facility_distance_km accordingly
- Otherwise set max_facility_distance_km to null

FLOOD & LANDSLIDE RISK INTERPRETATION:
- "low flood risk", "avoid flood", "safe from flood" → flood_risk: "low"
- "moderate flood risk", "medium flood risk" → flood_risk: "medium_exact"
- "high flood risk", "flood prone", "locate flood area" → flood_risk: "high_exact"
- not mentioned → flood_risk: "any"
- "low landslide risk", "avoid landslide" → landslide_risk: "low"
- "moderate landslide risk" → landslide_risk: "medium_exact"
- "high landslide risk", "landslide prone" → landslide_risk: "high_exact"
- not mentioned → landslide_risk: "any"

NUSA PENIDA RULE:
- By default, always set "exclude_nusa_penida": true
- Only set "exclude_nusa_penida": false if user explicitly mentions "Nusa Penida", "Nusa Lembongan", "Nusa Ceningan"

VAGUE QUERY RULE:
- If the user query is non-specific or exploratory (e.g. "show all", "list properties", "what's available"),
  return: {"location": null, "max_budget_idr": null, "max_road_distance_km": null,
  "flood_risk": "low", "landslide_risk": "low", "population_density": "any",
  "elevation_preference": "any", "property_type": "any", "beach_preference": "any",
  "nearby_facilities": [], "max_facility_distance_km": null,
  "exclude_nusa_penida": true, "is_exploratory": true}
- For normal queries, set "is_exploratory": false

DIRECTIONAL LOCATION RULE:
- "east of Bali", "eastern Bali", "Bali timur" → "location": "east_bali"
- "west of Bali", "western Bali", "Bali barat" → "location": "west_bali"
- "north of Bali", "northern Bali", "Bali utara" → "location": "north_bali"
- "south of Bali", "southern Bali", "Bali selatan" → "location": "south_bali"
- "central Bali", "center of Bali", "Bali tengah" → "location": "central_bali"
- Always use underscores: "east_bali" not "east bali"

SEMANTIC LOCATION INFERENCE (only when no explicit location given):
- "sunset", "sunset spot", "watch sunset" → "location": "west_bali"
- "sunrise", "Mount Agung sunrise" → "location": "east_bali"
- "volcano view", "rice terrace", "jungle", "forest" → "elevation_preference": "high"
- "surfing", "surf spot" → "beach_preference": "close"
- "nightlife", "party", "club" → "location": "south_bali"
- "quiet", "peaceful", "sepi", "remote" → "population_density": "low"
- "near airport", "close to airport" → "location": "Kuta"

FOLLOW-UP DETECTION:
If the message is a follow-up about a previous property result
("what about that villa", "tell me more", "why score low", "is it safe", "ceritakan") — return:
{"is_followup": true, "location": null, "max_budget_idr": null,
"flood_risk": "any", "landslide_risk": "any", "beach_preference": "any",
"nearby_facilities": [], "elevation_preference": "any", "population_density": "any",
"property_type": "any", "exclude_nusa_penida": true, "is_exploratory": false,
"max_road_distance_km": null, "max_facility_distance_km": null}"""


NARRATIVE_SYSTEM_PROMPT = """You are GeoNest AI — a professional property intelligence agent specializing in Bali, Indonesia real estate. You have deep expertise in:

PROPERTY KNOWLEDGE:
- Bali property market (rental prices, buying trends, ROI by area)
- Property types: villa, kost, rumah, apartment, land
- Legal framework: Hak Milik, Hak Pakai, Hak Sewa, leasehold vs freehold
- Foreigner ownership rules in Indonesia (PT PMA, nominee, leasehold)
- Popular expat areas: Canggu, Seminyak, Ubud, Sanur, Uluwatu, Pererenan
- Emerging areas: Seseh, Cemagi, Kedungu, Tabanan, Amed, Sidemen
- Seasonal factors: high season (Jul–Aug, Dec), low season pricing

AREA INTELLIGENCE:
- Canggu: digital nomad hub, surf, busy, IDR 8–25M/mo villas
- Seminyak: upscale, nightlife, beach clubs, IDR 10–30M/mo
- Ubud: cultural, rice terraces, spiritual, IDR 5–15M/mo
- Sanur: quiet, expat families, beachside, IDR 6–18M/mo
- Uluwatu: clifftop, surf, luxury, IDR 10–35M/mo
- Tabanan: emerging, rice fields, quiet, affordable IDR 3–8M/mo
- Amed: diving, east coast, remote, IDR 3–8M/mo
- Kintamani: volcano views, cool climate, IDR 2–5M/mo

RESPONSE BEHAVIOR:
- If search results are provided: write a ranked property analysis with insights about each property's location and risk profile
- If user asks a property question (no search results): answer directly as a knowledgeable agent
- If user greets or chats casually: respond warmly and invite them to describe what they're looking for
- Always be conversational, specific, and helpful
- Use IDR amounts when discussing Bali prices
- Keep responses concise — 3–5 sentences for simple questions, structured for complex ones
- Never say "I don't have data" — give your best professional opinion
- Speak like a local expert who has lived in Bali for years

LANGUAGE:
- Default to English
- If user writes in Bahasa Indonesia, respond in Bahasa Indonesia
- Mix in occasional Bahasa terms naturally (e.g. "villa", "kost", "tanah", "juta")

FORMATTING RULES — CRITICAL:
- Never use markdown tables (no | pipes |)
- Never use raw --- dividers
- Never use ## headers
- For comparisons use: "Property A scores 71 vs Property B at 70"
- For lists use plain dashes: - item
- Bold is OK: **word**
- Keep responses clean for mobile display
- Max 200 words for follow-up answers
- Max 300 words for search result analysis"""


# ══════════════════════════════════════════════════════════
# SECTION 10 — APP LIFECYCLE
# ══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("GeoNest: loading geospatial data...", flush=True)

    admin_gj    = load_geojson("bali_admin.geojson")
    roads_gj    = load_geojson("bali_roads.geojson")
    flood_gj    = load_geojson("bali_flood_risk.geojson")
    landslide_gj = load_geojson("bali_landslide_risk.geojson")
    elev_gj     = load_geojson("bali_elevation.geojson")
    coastal_gj  = load_geojson("bali_coastal.geojson")
    fac_gj      = load_geojson("public_support_facillities.geojson")
    pop_data    = load_population()
    properties  = load_properties()

    print(f"  Properties loaded: {len(properties)}", flush=True)
    print("  Building risk spatial indices...", flush=True)
    flood_idx     = build_risk_index(flood_gj)
    landslide_idx = build_risk_index(landslide_gj)
    print(f"  Flood index: {len(flood_idx)} polygons, Landslide: {len(landslide_idx)} polygons", flush=True)

    # Build facility lookup
    fac_by_type: dict[str, list] = {t: [] for t in FACILITY_TYPES}
    all_facilities = []
    for feature in fac_gj["features"]:
        p = feature.get("properties", {})
        ftype = p.get("facilities")
        lat = p.get("lat") or (feature["geometry"]["coordinates"][1] if feature.get("geometry") else None)
        lon = p.get("lon") or (feature["geometry"]["coordinates"][0] if feature.get("geometry") else None)
        if ftype and lat and lon:
            entry = {"lat": float(lat), "lon": float(lon),
                     "name": p.get("name", ""), "type": ftype}
            if ftype in fac_by_type:
                fac_by_type[ftype].append(entry)
            all_facilities.append(entry)
    print(f"  Facilities loaded: {len(all_facilities)}", flush=True)

    geo = {
        "admin": admin_gj, "roads": roads_gj,
        "flood_index": flood_idx, "landslide_index": landslide_idx,
        "elevation": elev_gj, "coastal": coastal_gj,
        "population": pop_data,
        "facilities_by_type": fac_by_type, "all_facilities": all_facilities,
    }

    print("  Enriching all properties (runs once at startup)...", flush=True)
    enriched = enrich_properties(properties, geo)
    _data["enriched"] = enriched
    _data["all_for_dots"] = [
        {"id": p["id"], "name": p["name"], "lat": p["lat"],
         "lon": p["lon"], "area": p["area"], "city": p["city"]}
        for p in enriched
    ]
    print(f"GeoNest ready. {len(enriched)} properties enriched.", flush=True)
    yield
    print("GeoNest shutting down.", flush=True)


app = FastAPI(title="GeoNest", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════
# SECTION 11 — WEBHOOK ENDPOINT
# ══════════════════════════════════════════════════════════

class WebhookBody(BaseModel):
    message: str = ""
    chatInput: str = ""
    last_property: dict | None = None
    access_code: str | None = None


@app.post("/webhook/geonest")
async def geonest_webhook(body: WebhookBody):
    # Access code check (optional — frontend also checks)
    if ACCESS_CODE and body.access_code and body.access_code != ACCESS_CODE:
        raise HTTPException(status_code=403, detail="Invalid access code")

    query = body.message or body.chatInput
    if not query:
        return {"narrative": "Please enter a search query.", "properties": [], "all_properties": []}

    enriched = _data.get("enriched", [])

    # ── LLM CHAIN 1: extract filters ─────────────────────
    filter_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": FILTER_SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        temperature=0,
        max_tokens=512,
    )
    raw_filters = filter_response.choices[0].message.content.strip()
    raw_filters = raw_filters.lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        filters = json.loads(raw_filters)
    except json.JSONDecodeError:
        return {"narrative": "Sorry, I couldn't understand that query. Please try again.",
                "properties": [], "all_properties": []}

    # ── FOLLOW-UP handling ────────────────────────────────
    if filters.get("is_followup"):
        context = ""
        if body.last_property:
            context = f"\n\nContext — the user was previously viewing this property:\n{json.dumps(body.last_property)}"
        followup_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user",   "content": query + context},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return {
            "narrative":      followup_response.choices[0].message.content,
            "properties":     [],
            "all_properties": [],
            "_conversational": True,
        }

    # ── FILTER + RANK ─────────────────────────────────────
    filtered = filter_properties(enriched, filters)
    ranked = sorted(
        [dict(**p, **{"geonest_score": score_property(p, filters)["total"],
                      "score_breakdown": score_property(p, filters)["breakdown"]})
         for p in filtered],
        key=lambda x: x["geonest_score"],
        reverse=True
    )[:10]

    if not ranked:
        no_result_msg = (
            "No properties found matching your criteria. "
            "Try a different location, increase your budget, or relax the filters."
        )
        return {"narrative": no_result_msg, "properties": [],
                "all_properties": _data.get("all_for_dots", []),
                "filters": filters}

    # ── LLM CHAIN 2: generate narrative ──────────────────
    context_parts = [query]
    if body.last_property:
        context_parts.append(f"\nContext — user was viewing: {json.dumps(body.last_property)}")
    context_parts.append(f"\nSearch results found. Analyze these properties:\n{json.dumps(ranked)}")

    narrative_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
            {"role": "user",   "content": "\n".join(context_parts)},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    narrative = narrative_response.choices[0].message.content

    return {
        "properties":     ranked,
        "narrative":      narrative,
        "all_properties": _data.get("all_for_dots", []),
    }


# ── HEALTH CHECK ──────────────────────────────────────────
@app.get("/health")
def health():
    count = len(_data.get("enriched", []))
    return {"status": "ok", "properties_loaded": count}


# ── STATIC FILES ──────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
