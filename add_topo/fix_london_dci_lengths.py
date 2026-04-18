"""
Fix LondonDCI edge lengths in MongoDB.

The topology was stored with all length_km=80 and span_length_km=80,
but the actual haversine distances between nodes are 1–28 km.

This script:
  1. Reads the LondonDCI document from Topology_Data.real
  2. Recalculates length_km and span_length_km from node lat/lon (haversine)
  3. Keeps weight=1 (one span per link, used as hop count for routing)
  4. Writes the corrected topology_data back to MongoDB

Run:
    python fix_london_dci_lengths.py [--host HOST] [--port PORT]
                                     [--user USER] [--password PWD]
                                     [--dry-run]
"""

import argparse
import ast
import math
from bson import ObjectId
from pymongo import MongoClient

# ── document coordinates ────────────────────────────────────────────────────
DB         = "Topology_Data"
COLLECTION = "real"
DOC_ID     = ObjectId("69c92385e8da37637e38d479")

EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_R_KM * 2 * math.asin(math.sqrt(a))


def build_corrected_topology_data(topology_data: dict, node_data: dict) -> dict:
    """Return a new topology_data dict with corrected length_km / span_length_km."""
    corrected = {}
    for src, neighbours in topology_data.items():
        corrected[src] = {}
        lat1 = float(node_data[src]["latitude"])
        lon1 = float(node_data[src]["longitude"])
        for dst, edge_str in neighbours.items():
            lat2 = float(node_data[dst]["latitude"])
            lon2 = float(node_data[dst]["longitude"])
            dist_km = round(haversine_km(lat1, lon1, lat2, lon2), 4)

            edge_dict = ast.literal_eval(edge_str)
            edge_dict["length_km"]      = dist_km
            edge_dict["span_length_km"] = dist_km  # one span per link
            # weight stays 1 (hop count used by kSP routing)
            corrected[src][dst] = str(edge_dict)
    return corrected


def main():
    parser = argparse.ArgumentParser(description="Fix LondonDCI edge lengths")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     default=27017, type=int)
    parser.add_argument("--user",     default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print changes without writing to MongoDB")
    args, _ = parser.parse_known_args()

    # ── connect ──────────────────────────────────────────────────────────────
    if args.user and args.password:
        uri = f"mongodb://{args.user}:{args.password}@{args.host}:{args.port}/"
    else:
        uri = f"mongodb://{args.host}:{args.port}/"
    client = MongoClient(uri)
    col = client[DB][COLLECTION]

    # ── fetch document ────────────────────────────────────────────────────────
    doc = col.find_one({"_id": DOC_ID})
    if doc is None:
        raise RuntimeError(f"Document {DOC_ID} not found in {DB}.{COLLECTION}")
    print(f"Found document: name='{doc.get('name')}', _id={doc['_id']}")

    topology_data = doc["topology data"]
    node_data     = doc["node data"]

    # ── compute corrected edges ───────────────────────────────────────────────
    corrected = build_corrected_topology_data(topology_data, node_data)

    # ── print diff ────────────────────────────────────────────────────────────
    print(f"\n{'Edge':<10} {'Old length_km':>14} {'New length_km':>14}")
    print("-" * 42)
    seen = set()
    for src, neighbours in corrected.items():
        for dst, edge_str in neighbours.items():
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            new_dist = ast.literal_eval(edge_str)["length_km"]
            old_dist = ast.literal_eval(topology_data[src][dst])["length_km"]
            marker = " ✓" if abs(new_dist - old_dist) < 0.01 else " ← CHANGED"
            print(f"({src:>2},{dst:>2})    {old_dist:>14}  {new_dist:>12.4f}{marker}")

    if args.dry_run:
        print("\n[dry-run] No changes written to MongoDB.")
        return

    # ── write back ────────────────────────────────────────────────────────────
    result = col.update_one(
        {"_id": DOC_ID},
        {"$set": {"topology data": corrected}},
    )
    print(f"\nMongoDB update: matched={result.matched_count}, "
          f"modified={result.modified_count}")
    print("Done. topology_data.length_km and span_length_km updated "
          "to haversine distances.")
    client.close()


if __name__ == "__main__":
    main()
