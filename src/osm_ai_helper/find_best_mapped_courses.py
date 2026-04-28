import json
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from fire import Fire
from loguru import logger
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _query_overpass(region: str, timeout: int = 180) -> List[Dict]:
    query = f"""
    [out:json][timeout:{timeout}];
    area["name:en"~"^{region}$",i]->.searchArea;
    area["name"~"^{region}$",i]->.searchAreaLocal;
    (
      way["leisure"="golf_course"](area.searchArea);
      way["leisure"="golf_course"](area.searchAreaLocal);
      relation["leisure"="golf_course"](area.searchArea);
      relation["leisure"="golf_course"](area.searchAreaLocal);
      way["golf"="bunker"](area.searchArea);
      way["golf"="bunker"](area.searchAreaLocal);
    );
    out geom;
    """
    response = requests.get(
        OVERPASS_URL,
        params={"data": query},
        headers={"User-Agent": "osm-ai-helper/find_best_mapped_courses"},
        timeout=timeout + 30,
    )
    response.raise_for_status()
    return response.json()["elements"]


def _course_polygon(course: Dict) -> Polygon | None:
    if course["type"] == "way":
        coords = [(p["lon"], p["lat"]) for p in course.get("geometry", [])]
        if len(coords) < 4:
            return None
        poly = Polygon(coords)
    elif course["type"] == "relation":
        outer_rings = [
            [(p["lon"], p["lat"]) for p in m.get("geometry", [])]
            for m in course.get("members", [])
            if m.get("role") == "outer"
            and m.get("type") == "way"
            and len(m.get("geometry", [])) >= 4
        ]
        if not outer_rings:
            return None
        poly = max((Polygon(r) for r in outer_rings), key=lambda p: p.area)
    else:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def _bunker_centroid(bunker: Dict) -> Point | None:
    coords = [(p["lon"], p["lat"]) for p in bunker.get("geometry", [])]
    if len(coords) < 3:
        return None
    try:
        return Polygon(coords).centroid
    except Exception:
        return None


@logger.catch(reraise=True)
def find_best_mapped_courses(
    output_dir: str,
    region: str = "United Kingdom",
    n_train: int = 8,
    n_val: int = 2,
    skip_course_ids: List[int] | None = None,
) -> Tuple[Path, Path, Path]:
    """Find the top-N best-mapped golf courses in a region by tagged bunker count.

    Queries OSM for all `leisure=golf_course` polygons and `golf=bunker` ways in
    the given region, assigns each bunker to the course polygon containing its
    centroid, ranks courses by bunker count, and writes train/val bunker element
    files in the same shape as `download_osm` (so the rest of the pipeline works
    unchanged).

    Use `skip_course_ids` to exclude specific courses from the ranking — for
    example, courses whose bunker mapping style is too distinct from the rest
    (huge dune-style bunkers vs. small pot bunkers) and would skew validation.

    Args:
        output_dir (str): Directory to write the output files.
        region (str): OSM area name (e.g. "United Kingdom", "Ireland",
            "Scotland"). Used as the search area for the Overpass query.
        n_train (int): Number of top courses to use for training.
        n_val (int): Number of next-ranked courses to use for validation.

    Returns:
        (train_file, val_file, summary_file) paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Querying Overpass for golf courses + bunkers in {region!r}")
    elements = _query_overpass(region)

    courses: List[Dict] = []
    bunkers: List[Dict] = []
    for elem in elements:
        tags = elem.get("tags", {}) or {}
        if tags.get("leisure") == "golf_course":
            courses.append(elem)
        elif tags.get("golf") == "bunker" and elem["type"] == "way":
            bunkers.append(elem)

    logger.info(f"Found {len(courses)} courses and {len(bunkers)} bunker ways")
    if not courses:
        raise ValueError(f"No golf courses found in region {region!r}")

    course_polys: List[Tuple[Dict, Polygon]] = []
    for course in courses:
        poly = _course_polygon(course)
        if poly is not None:
            course_polys.append((course, poly))
    logger.info(f"Built {len(course_polys)} valid course polygons")

    polys = [p for _, p in course_polys]
    prepared = [prep(p) for p in polys]
    tree = STRtree(polys)

    bunkers_per_course: List[List[Dict]] = [[] for _ in course_polys]
    unassigned = 0
    for bunker in bunkers:
        centroid = _bunker_centroid(bunker)
        if centroid is None:
            continue
        candidate_idxs = tree.query(centroid)
        assigned = False
        for idx in candidate_idxs:
            if prepared[idx].contains(centroid):
                bunkers_per_course[idx].append(bunker)
                assigned = True
                break
        if not assigned:
            unassigned += 1

    logger.info(
        f"Assigned {sum(len(b) for b in bunkers_per_course)} bunkers to courses; "
        f"{unassigned} bunkers fell outside any course polygon (ignored)"
    )

    skip_set = set(skip_course_ids or [])
    ranked_idxs = sorted(
        range(len(course_polys)),
        key=lambda i: -len(bunkers_per_course[i]),
    )
    if skip_set:
        excluded = [i for i in ranked_idxs if course_polys[i][0]["id"] in skip_set]
        for i in excluded:
            course = course_polys[i][0]
            name = (course.get("tags") or {}).get("name") or "(unnamed)"
            logger.info(f"Excluding course {name} ({course['type']}/{course['id']}) per skip_course_ids")
        ranked_idxs = [i for i in ranked_idxs if course_polys[i][0]["id"] not in skip_set]
    top = ranked_idxs[: n_train + n_val]

    summary = []
    train_bunkers: List[Dict] = []
    val_bunkers: List[Dict] = []
    for rank, idx in enumerate(top):
        course, _ = course_polys[idx]
        course_bunkers = bunkers_per_course[idx]
        split = "train" if rank < n_train else "val"
        info = {
            "rank": rank + 1,
            "split": split,
            "osm_type": course["type"],
            "osm_id": course["id"],
            "name": (course.get("tags") or {}).get("name"),
            "bunker_count": len(course_bunkers),
        }
        summary.append(info)
        logger.info(
            f"#{rank + 1} [{split}] {info['name'] or '(unnamed)'} "
            f"({info['osm_type']}/{info['osm_id']}) — {info['bunker_count']} bunkers"
        )
        if split == "train":
            train_bunkers.extend(course_bunkers)
        else:
            val_bunkers.extend(course_bunkers)

    train_file = output_path / "courses_train.json"
    val_file = output_path / "courses_val.json"
    summary_file = output_path / "courses_summary.json"
    train_file.write_text(json.dumps(train_bunkers))
    val_file.write_text(json.dumps(val_bunkers))
    summary_file.write_text(json.dumps(summary, indent=2))

    logger.success(
        f"Wrote {len(train_bunkers)} train bunkers, {len(val_bunkers)} val bunkers"
    )
    return train_file, val_file, summary_file


if __name__ == "__main__":
    Fire(find_best_mapped_courses)
