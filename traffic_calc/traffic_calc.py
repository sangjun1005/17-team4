"""천안시 읍·면·동 교통 취약성 계산 모듈.

이 모듈은 교통 하위지표를 두 가지로만 계산한다.
1) 읍·면·동별 일일 운행 공급량(노선-정류장 기준)
2) 읍·면·동별 정류장 밀도(고유 정류장 수 / 면적)

실제 승객 수요나 승하차량이 아니라, 노선 자료에 기록된 예정 운행횟수를
사용하므로 결과의 명칭은 '운행 공급량'으로 해석해야 한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROUTES = BASE_DIR / "route_stops_cheonan.csv"
DEFAULT_STOPS = BASE_DIR / "천안시_버스정류장_천안시만_20251031.csv"
DEFAULT_ROUTE_MASTER = BASE_DIR / "천안시_시내버스운수업체별노선현황_20260310.csv"
DEFAULT_AREAS = BASE_DIR / "cheonan-subdivisions.json"
DEFAULT_OUTPUT = BASE_DIR / "traffic_vulnerability_scores.csv"


def read_csv_rows(path: Path) -> Iterable[dict]:
    """UTF-8 BOM 또는 CP949 CSV를 읽는다."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                yield from csv.DictReader(handle)
            return
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error


def parse_number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def ring_area_km2(ring: Sequence[Sequence[float]]) -> float:
    """경위도 링을 지역 평면으로 투영해 면적(km²)을 계산한다."""
    if len(ring) < 3:
        return 0.0
    lat0 = math.radians(sum(float(point[1]) for point in ring) / len(ring))
    radius_km = 6371.0088
    points = [
        (
            radius_km * math.cos(lat0) * math.radians(float(point[0])),
            radius_km * math.radians(float(point[1])),
        )
        for point in ring
    ]
    signed = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        signed += x1 * y2 - x2 * y1
    return abs(signed) / 2.0


def geometry_area_km2(geometry: dict) -> float:
    """Polygon/MultiPolygon의 외곽 링 면적에서 구멍 면적을 뺀다."""
    if not geometry:
        return 0.0
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    polygons = coordinates if geometry_type == "MultiPolygon" else [coordinates]
    total = 0.0
    for polygon in polygons:
        if not polygon:
            continue
        outer = ring_area_km2(polygon[0])
        holes = sum(ring_area_km2(ring) for ring in polygon[1:])
        total += max(0.0, outer - holes)
    return total


def load_area_features(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        document = json.load(handle)
    areas: Dict[str, dict] = {}
    for feature in document.get("features", []):
        props = feature.get("properties", {})
        name = str(props.get("name", "")).strip()
        if not name:
            continue
        areas[name] = {
            "구": str(props.get("district", "")).strip(),
            "유형": str(props.get("kind", "")).strip(),
            "면적_km2": geometry_area_km2(feature.get("geometry", {})),
        }
    return areas


def stop_key(row: dict) -> str:
    """정류장 코드 우선, 없으면 대체 식별자를 사용한다."""
    for field in ("정류장코드", "정류장번호_full", "정류장명_마스터", "정류장명_크롤링"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def location_stop_key(row: dict) -> str:
    """위치정보 자료의 정류장 식별자."""
    for field in ("정류장번호", "모바일단축번호"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    name = str(row.get("정류장명", "")).strip()
    lat = str(row.get("위도", "")).strip()
    lon = str(row.get("경도", "")).strip()
    return f"{name}|{lat}|{lon}" if name or lat or lon else ""


def aggregate_stop_locations(path: Path) -> Tuple[defaultdict[str, set], set]:
    """천안시 정류장 위치정보를 읍면동별로 고유 집계한다."""
    unique_stops: defaultdict[str, set] = defaultdict(set)
    dates: set[str] = set()
    for row in read_csv_rows(path):
        area = str(row.get("읍면동", "")).strip()
        key = location_stop_key(row)
        if not area or not key:
            continue
        unique_stops[area].add(key)
        date = str(row.get("정보수집일", "")).strip()
        if date:
            dates.add(date)
    return unique_stops, dates


def load_route_master(path: Path) -> Tuple[Dict[str, float], set]:
    """노선현황 원본의 노선별 1일 운행횟수와 기준일을 읽는다."""
    frequencies: Dict[str, float] = {}
    dates: set[str] = set()
    for row in read_csv_rows(path):
        route = str(row.get("노선번호", "")).strip()
        if not route:
            continue
        frequencies[route] = parse_number(row.get("1일운행횟수(회_편도기준)"))
        date = str(row.get("데이터기준일자", "")).strip()
        if date:
            dates.add(date)
    return frequencies, dates


def aggregate_routes(path: Path, master_path: Path) -> Tuple[dict, set, set, set]:
    """읍면동별 고유 정류장·일일 운행 공급량을 집계한다.

    동일 노선-정류장 행이 중복으로 들어온 경우를 막기 위해
    (읍면동, 노선번호, 정류장 식별자)를 한 번만 반영한다.
    """
    unique_stops: defaultdict[str, set] = defaultdict(set)
    daily_supply: defaultdict[str, float] = defaultdict(float)
    seen_route_stops: set[tuple[str, str, str]] = set()
    dates: set[str] = set()
    all_areas: set[str] = set()
    master_frequencies, master_dates = load_route_master(master_path)
    missing_master_routes: set[str] = set()

    for row in read_csv_rows(path):
        area = str(row.get("읍면동", "")).strip()
        if not area:
            continue
        all_areas.add(area)
        route = str(row.get("노선번호", "")).strip()
        key = stop_key(row)
        if not key:
            continue
        dedupe_key = (area, route, key)
        if dedupe_key in seen_route_stops:
            continue
        seen_route_stops.add(dedupe_key)
        unique_stops[area].add(key)
        if route in master_frequencies:
            daily_supply[area] += master_frequencies[route]
        else:
            daily_supply[area] += parse_number(row.get("1일운행횟수"))
            missing_master_routes.add(route)
        date = str(row.get("데이터기준일자", "")).strip()
        if date:
            dates.add(date)

    return (
        {"unique_stops": unique_stops, "daily_supply": daily_supply},
        all_areas,
        dates | master_dates,
        missing_master_routes,
    )


def inverse_minmax(values: Dict[str, float]) -> Dict[str, float]:
    """낮은 원자료일수록 취약점수가 커지는 0~100 역 min-max 점수."""
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if math.isclose(minimum, maximum):
        return {key: 50.0 for key in values}
    return {
        key: 100.0 * (1.0 - (value - minimum) / (maximum - minimum))
        for key, value in values.items()
    }


def round_value(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def calculate(
    routes_path: Path,
    stops_path: Path,
    route_master_path: Path,
    areas_path: Path,
    output_path: Path,
) -> List[dict]:
    area_info = load_area_features(areas_path)
    aggregates, route_areas, dates, missing_master_routes = aggregate_routes(
        routes_path, route_master_path
    )
    route_unique_stops = aggregates["unique_stops"]
    unique_stops, stop_dates = aggregate_stop_locations(stops_path)
    daily_supply = aggregates["daily_supply"]

    # 경계 GeoJSON의 31개 행정구역을 기준으로 하므로 노선 자료에 없는 구역도 0으로 남긴다.
    names = sorted(area_info)
    stop_counts = {name: float(len(unique_stops.get(name, set()))) for name in names}
    supply_values = {name: float(daily_supply.get(name, 0.0)) for name in names}
    density_values = {
        name: stop_counts[name] / area_info[name]["면적_km2"]
        if area_info[name]["면적_km2"] > 0
        else 0.0
        for name in names
    }
    supply_shortage = inverse_minmax(supply_values)
    density_shortage = inverse_minmax(density_values)

    ranked = []
    for name in names:
        score = 0.5 * supply_shortage[name] + 0.5 * density_shortage[name]
        ranked.append(
            {
                "읍면동": name,
                "구": area_info[name]["구"],
                "유형": area_info[name]["유형"],
                "면적_km2": round_value(area_info[name]["면적_km2"]),
                "고유정류장수": int(stop_counts[name]),
                "노선자료내고유정류장수": int(len(route_unique_stops.get(name, set()))),
                "일일운행공급량": round_value(supply_values[name]),
                "정류장밀도_개_per_km2": round_value(density_values[name]),
                "운행공급부족도_점수": round_value(supply_shortage[name], 2),
                "정류장밀도부족도_점수": round_value(density_shortage[name], 2),
                "교통취약점수": round_value(score, 2),
                "면적산출방식": "GeoJSON 경계 좌표 계산",
                "운행자료기준일자": ";".join(sorted(dates)),
                "정류장위치자료기준일자": ";".join(sorted(stop_dates)),
            }
        )
    ranked.sort(key=lambda row: (-row["교통취약점수"], row["읍면동"]))
    for index, row in enumerate(ranked, start=1):
        row["교통취약순위"] = index

    fieldnames = [
        "읍면동",
        "구",
        "유형",
        "면적_km2",
        "고유정류장수",
        "노선자료내고유정류장수",
        "일일운행공급량",
        "정류장밀도_개_per_km2",
        "운행공급부족도_점수",
        "정류장밀도부족도_점수",
        "교통취약점수",
        "교통취약순위",
        "면적산출방식",
        "운행자료기준일자",
        "정류장위치자료기준일자",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    missing_route_areas = sorted(set(area_info) - route_areas)
    if missing_route_areas:
        print("노선 자료가 없는 행정구역(0으로 계산):", ", ".join(missing_route_areas))
    if missing_master_routes:
        print(
            "노선현황 원본에 없어 route-stop 운행횟수를 fallback으로 사용한 노선:",
            ", ".join(sorted(missing_master_routes)),
        )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="천안시 교통 취약성(두 지표) 계산")
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES, help="노선-정류장 CSV")
    parser.add_argument("--stops", type=Path, default=DEFAULT_STOPS, help="천안시 정류장 위치정보 CSV")
    parser.add_argument(
        "--route-master",
        type=Path,
        default=DEFAULT_ROUTE_MASTER,
        help="노선별 운행횟수 원본 CSV",
    )
    parser.add_argument("--areas", type=Path, default=DEFAULT_AREAS, help="천안 행정구역 GeoJSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="결과 CSV")
    args = parser.parse_args()
    rows = calculate(args.routes, args.stops, args.route_master, args.areas, args.output)
    print(f"행정구역 수: {len(rows)}")
    print(f"결과 파일: {args.output.resolve()}")
    print("점수 구성: 운행공급부족도 50% + 정류장밀도부족도 50%")


if __name__ == "__main__":
    main()
