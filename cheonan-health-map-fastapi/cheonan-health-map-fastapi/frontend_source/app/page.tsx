"use client";

import { useEffect, useRef, useState } from "react";
import { CHEONAN_BOUNDARY } from "./cheonan-boundary";
import CHEONAN_SUBDIVISIONS from "./cheonan-subdivisions.json";
import {
  getVulnerabilitySummary,
  VULNERABILITY_MODELS,
  type WeightingModel,
} from "./vulnerability-summary";
import {
  ALL_REGION_CODES,
  isPilotRegionCode,
  PILOT_REGION_CODES,
  REGION_BY_NAME,
  REGION_CODE_BY_NAME,
  regionName,
  type RegionKind,
  type RegionProperties,
} from "./region-catalog";
import { REGION_DASHBOARD_DATA } from "./region-dashboard-data";
import { REGION_SUPPLEMENTAL_BY_NAME } from "./region-supplemental-data";
import { CHEONAN_AGE_BY_NAME } from "./cheonan-age-population";

type AreaCode = string;
type PilotAreaName = "광덕면" | "풍세면" | "청룡동" | "신방동";
type ComparisonMetric = "elderly-count" | "elderly-rate" | "age-band" | "living-alone" | "disabled";
type ComparisonScope = "all" | "top10";
type ComparisonSort = "rank" | "name" | "elderly";
type SubdivisionKind = RegionKind;
type SubdivisionProperties = RegionProperties;

type AreaProfile = {
  tone: "critical" | "watch";
  label: string;
  medicalInstitutions?: number | null;
  publicInstitutions?: number | null;
  privateTotal?: number | null;
  clinic?: number | null;
  dental?: number | null;
  oriental?: number | null;
  pharmacy?: number | null;
  healthBranch?: number | null;
  healthClinic?: number | null;
  verified: Array<{
    type: string;
    name: string;
    address: string;
    phone: string;
  }>;
};

const SUBDIVISION_BY_NAME: Record<string, SubdivisionProperties> = REGION_BY_NAME;

const pilotName = (code: AreaCode) => regionName(code);

const KIND_PALETTE: Record<SubdivisionKind, { stroke: string; fill: string }> = {
  읍: { stroke: "#2563eb", fill: "#dbeafe" },
  면: { stroke: "#16a34a", fill: "#dcfce7" },
  동: { stroke: "#7c3aed", fill: "#ede9fe" },
};

const POPULATION_DATA = Object.fromEntries(
  Object.entries(REGION_DASHBOARD_DATA).map(([code, row]) => {
    const supplemental = REGION_SUPPLEMENTAL_BY_NAME[row.name];
    const ageData = CHEONAN_AGE_BY_NAME[row.name];
    return [
      code,
      {
        total: row.total,
        elderly: row.elderly,
        elderlyRate: row.elderlyRate,
        ageBands: row.ageBands ?? ageData?.ageBands ?? null,
        ageComposition: row.ageComposition ?? ageData?.ageComposition ?? null,
        livingAlone: row.livingAlone,
        basicLiving: row.basicLiving ?? supplemental?.basicLiving ?? null,
        lowIncome: row.lowIncome ?? supplemental?.lowIncome ?? null,
        disabled: row.disabled,
      },
    ];
  }),
) as Record<AreaCode, {
  total: number;
  elderly: number;
  elderlyRate: number;
  ageBands: number[] | null;
  ageComposition: number[] | null;
  livingAlone: number | null;
  basicLiving: number | null;
  lowIncome: number | null;
  disabled: number | null;
}>;

const AREA_DATA = Object.fromEntries(
  Object.entries(REGION_DASHBOARD_DATA).map(([code, row]) => [
    code,
    {
      areaKm2: row.areaKm2 ?? 0,
      source: "천안시 행정구역 GeoJSON 경계 계산 · 2026.03.10",
    },
  ]),
) as Record<AreaCode, { areaKm2: number; source: string }>;

const AGE_COMPOSITION_LABELS = ["65~74세", "75~84세", "85세 이상"];
const AGE_COMPOSITION_COLORS = ["#2563eb", "#7c3aed", "#f59e0b"];
const COMPARISON_METRICS: Record<ComparisonMetric, string> = {
  "elderly-count": "65세 이상 인구수",
  "elderly-rate": "전체 인구 대비 65세 이상 비율",
  "age-band": "65세 이상 연령구간",
  "living-alone": "독거노인 구성",
  disabled: "등록장애인 수·비율",
};

const COMPARISON_SORT_LABELS: Record<ComparisonSort, string> = {
  rank: "취약순위순",
  name: "가나다순",
  elderly: "65세 이상 인구순",
};

const AGE_BAND_LABELS = ["65–69세", "70–74세", "75–79세", "80세 이상"];

type TransportProfile = {
  stops: number;
  sourceDate: string;
  method: string;
};

const TRANSPORT_DATA = Object.fromEntries(
  Object.entries(REGION_DASHBOARD_DATA).map(([code, row]) => [
    code,
    {
      stops: row.stops ?? 0,
      sourceDate: "2025.10.31",
      method: "천안시 행정구역 내부 고유 정류장번호 집계 · 미정차 제외",
    },
  ]),
) as Record<AreaCode, TransportProfile>;

function minMaxScore(value: number, values: number[]) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return 50;
  return ((value - min) / (max - min)) * 100;
}

function getTransportMetrics(area: AreaCode) {
  const elderly = POPULATION_DATA[area].elderly;
  const supply = (TRANSPORT_DATA[area].stops / elderly) * 1000;
  const rates = Object.keys(POPULATION_DATA).map(
    (name) => POPULATION_DATA[name].elderlyRate,
  );
  const supplies = Object.keys(POPULATION_DATA).map(
    (name) => (TRANSPORT_DATA[name].stops / POPULATION_DATA[name].elderly) * 1000,
  );
  const elderlyVulnerability = minMaxScore(POPULATION_DATA[area].elderlyRate, rates);
  const supplyQuality = minMaxScore(supply, supplies);
  const supplyShortage = 100 - supplyQuality;

  return {
    supply,
    elderlyVulnerability,
    supplyShortage,
    priority: (elderlyVulnerability + supplyShortage) / 2,
  };
}

function getElderlyDensity(area: AreaCode) {
  return POPULATION_DATA[area].elderly / AREA_DATA[area].areaKm2;
}

const CHEONAN_CENTER: [number, number] = [36.8151, 127.1139];
const formatCount = (value: number | null) => value === null ? "자료 없음" : value.toLocaleString();
const formatWeight = (value: number) => `${(value * 100).toFixed(value < 0.4 ? 1 : 0)}%`;
const PILOT_AREAS_BY_NAME: Record<PilotAreaName, AreaProfile> = {
  광덕면: {
    tone: "critical",
    label: "의료 접근 취약",
    privateTotal: 0,
    clinic: 0,
    dental: 0,
    oriental: 0,
    pharmacy: 0,
    healthBranch: 1,
    healthClinic: 2,
    verified: [
      {
        type: "보건지소",
        name: "광덕보건지소",
        address: "천안시 동남구 광덕면 신흥리3길 33",
        phone: "041-521-2677",
      },
      {
        type: "보건진료소",
        name: "행정보건진료소",
        address: "천안시 동남구 광덕면 차령고개로 1017",
        phone: "041-566-6690",
      },
      {
        type: "보건진료소",
        name: "보산원보건진료소",
        address: "천안시 동남구 광덕면 외보1길 17",
        phone: "041-563-1665",
      },
    ],
  },
  풍세면: {
    tone: "watch",
    label: "의료 접근 관찰",
    privateTotal: 3,
    clinic: 1,
    dental: 1,
    oriental: 1,
    pharmacy: 4,
    healthBranch: 1,
    healthClinic: 2,
    verified: [
      {
        type: "의원",
        name: "의)창신의료재단큰사랑의원",
        address: "천안시 동남구 풍세면 풍년길 41",
        phone: "041-566-8507",
      },
      {
        type: "보건지소",
        name: "풍세보건지소",
        address: "천안시 동남구 풍세면 상정1길 9",
        phone: "041-521-2576",
      },
      {
        type: "치과의원",
        name: "한양수치과의원",
        address: "천안시 동남구 풍세면 풍세산단로 287, 2층 203호",
        phone: "041-900-2877",
      },
      {
        type: "한의원",
        name: "생생한의원",
        address: "천안시 동남구 풍세면 풍세산단로 287, 2층 204호",
        phone: "041-522-5575",
      },
      {
        type: "약국",
        name: "남관약국",
        address: "천안시 동남구 풍세면 풍세로 465, 1층",
        phone: "041-571-1579",
      },
      {
        type: "약국",
        name: "수자인약국",
        address: "천안시 동남구 풍세면 풍세산단로 290, 상가동 113호",
        phone: "041-556-6256",
      },
      {
        type: "약국",
        name: "풍세서울약국",
        address: "천안시 동남구 풍세면 풍년길 46-1",
        phone: "041-551-8002",
      },
      {
        type: "약국",
        name: "풍세일번약국",
        address: "천안시 동남구 풍세면 풍세산단로 287, 1층 105호",
        phone: "041-523-4947",
      },
      {
        type: "보건진료소",
        name: "미죽보건진료소",
        address: "천안시 동남구 풍세면 미죽2길 6",
        phone: "041-566-6881",
      },
      {
        type: "보건진료소",
        name: "용정보건진료소",
        address: "천안시 동남구 풍세면 돈마루1길 29",
        phone: "041-566-7796",
      },
    ],
  },
  청룡동: {
    tone: "watch",
    label: "의료자원 통계 확인",
    // 천안시 기본통계 2024년 읍면동별 병원·약국 집계
    privateTotal: 49,
    clinic: 21,
    dental: 17,
    oriental: 11,
    pharmacy: 22,
    healthBranch: 0,
    healthClinic: 0,
    verified: [],
  },
  신방동: {
    tone: "watch",
    label: "의료자원 통계 확인",
    // 천안시 기본통계 2024년 읍면동별 병원·약국 집계
    privateTotal: 41,
    clinic: 21,
    dental: 10,
    oriental: 10,
    pharmacy: 20,
    healthBranch: 0,
    healthClinic: 0,
    verified: [],
  },
};

const AREA_PROFILES = Object.fromEntries(
  ALL_REGION_CODES.map((code) => {
    const row = REGION_DASHBOARD_DATA[code];
    const detailed = PILOT_AREAS_BY_NAME[row.name];
    return [
      code,
      {
        tone: detailed?.tone ?? "watch",
        label: detailed?.label ?? "지역 통계 확인",
        medicalInstitutions: row.medicalInstitutions,
        publicInstitutions: row.publicInstitutions,
        pharmacy: detailed?.pharmacy ?? REGION_SUPPLEMENTAL_BY_NAME[row.name]?.pharmacies ?? null,
        verified: detailed?.verified ?? [],
      },
    ];
  }),
) as Record<AreaCode, AreaProfile & { medicalInstitutions: number | null; publicInstitutions: number | null; pharmacy: number | null }>;

function isPilotArea(name: string) {
  const code = REGION_CODE_BY_NAME[name];
  return code ? isPilotRegionCode(code) : false;
}

function vulnerabilityPalette(rank: number) {
  if (rank <= 5) return { fill: "#dc2626", stroke: "#991b1b" };
  if (rank <= 10) return { fill: "#f97316", stroke: "#c2410c" };
  if (rank <= 20) return { fill: "#facc15", stroke: "#a16207" };
  return { fill: "#86efac", stroke: "#15803d" };
}

function areaStyle(name: string, selected: AreaCode | null, weightingModel: WeightingModel) {
  const subdivision = SUBDIVISION_BY_NAME[name];
  const kind = subdivision?.kind;
  const palette = kind ? KIND_PALETTE[kind] : { stroke: "#94a3b8", fill: "#cbd5e1" };
  const code = REGION_CODE_BY_NAME[name];
  const active = selected === code;
  const rank = code ? getVulnerabilitySummary(code, weightingModel).rank : Number.POSITIVE_INFINITY;
  const urgency = Number.isFinite(rank) ? vulnerabilityPalette(rank) : palette;

  return {
    // 선택 영역은 사각형 bounds가 아니라 해당 행정구역의 실제 GeoJSON 외곽선을 강조한다.
    color: active ? "#111827" : urgency.stroke,
    // 시범지역만 강조하고, 나머지 27개 경계는 보조선으로 낮춰 겹침을 줄인다.
    weight: active ? 4 : 1.2,
    opacity: active ? 1 : 0.88,
    fillColor: urgency.fill,
    fillOpacity: active ? 0.58 : 0.32,
  };
}

export default function Home() {
  const mapElement = useRef<HTMLDivElement>(null);
  const leftRail = useRef<HTMLElement | null>(null);
  const detailCard = useRef<HTMLElement | null>(null);
  const mapLegend = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<import("leaflet").Map | null>(null);
  const cityBounds = useRef<import("leaflet").LatLngBounds | null>(null);
  const areaLayers = useRef<Record<string, import("leaflet").Path>>({});
  const areaBounds = useRef<Record<AreaCode, import("leaflet").LatLngBounds>>({});
  const fitCityToViewport = useRef<(animate: boolean) => void>(() => {});
  const focusAreaInViewport = useRef<(area: AreaCode) => void>(() => {});
  const activeMapArea = useRef<AreaCode | null>(null);
  const mapFocusModeRef = useRef(false);
  const [selectedArea, setSelectedArea] = useState<AreaCode>(PILOT_REGION_CODES[0]);
  const [weightingModel, setWeightingModel] = useState<WeightingModel>("equal");
  const [comparisonMetric, setComparisonMetric] = useState<ComparisonMetric>("elderly-count");
  const [comparisonScope, setComparisonScope] = useState<ComparisonScope>("all");
  const [comparisonSort, setComparisonSort] = useState<ComparisonSort>("rank");
  const [mapReady, setMapReady] = useState(false);
  const [mapFocusMode, setMapFocusMode] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [comparisonOpen, setComparisonOpen] = useState(false);

  useEffect(() => {
    let disposed = false;

    function cleanupMap() {
      mapInstance.current?.remove();
      mapInstance.current = null;
      areaLayers.current = {};
      areaBounds.current = {};
      fitCityToViewport.current = () => {};
      focusAreaInViewport.current = () => {};
      activeMapArea.current = null;

      // Fast Refresh가 이전 Leaflet 인스턴스를 놓친 경우에도
      // 기존 SVG/Canvas 레이어와 컨테이너 식별자를 남기지 않습니다.
      const mapHost = mapElement.current as (HTMLDivElement & { _leaflet_id?: number }) | null;
      if (mapHost) {
        delete mapHost._leaflet_id;
        mapHost.replaceChildren();
        mapHost.className = "map";
        mapHost.removeAttribute("style");
      }
    }

    async function createMap() {
      cleanupMap();
      if (!mapElement.current) return;
      const L = await import("leaflet");
      if (disposed || !mapElement.current) return;

      const boundaryLatLngs = CHEONAN_BOUNDARY.map(([lng, lat]) =>
        L.latLng(lat, lng),
      );
      const boundaryBounds = L.latLngBounds(boundaryLatLngs);
      const map = L.map(mapElement.current, {
        center: CHEONAN_CENTER,
        zoom: 11,
        maxZoom: 18,
        maxBounds: boundaryBounds.pad(0.08),
        maxBoundsViscosity: 1,
        boxZoom: false,
        zoomSnap: 0.25,
        zoomControl: false,
        attributionControl: true,
      });

      const resolveMapPadding = () => {
        const hostBounds = mapElement.current?.getBoundingClientRect();
        if (!hostBounds || mapFocusModeRef.current) {
          return {
            paddingTopLeft: [28, 28] as [number, number],
            paddingBottomRight: [28, 28] as [number, number],
          };
        }

        const visibleBounds = (element: Element | null) => {
          const bounds = element?.getBoundingClientRect();
          return bounds && bounds.width > 0 && bounds.height > 0 ? bounds : null;
        };
        const leftBounds = visibleBounds(leftRail.current);
        const detailBounds = visibleBounds(detailCard.current);
        const legendBounds = visibleBounds(mapLegend.current);
        const headerBounds = visibleBounds(document.querySelector(".map-header"));

        return {
          paddingTopLeft: [
            Math.max(28, (leftBounds?.right ?? hostBounds.left) - hostBounds.left + 18),
            Math.max(28, (headerBounds?.bottom ?? hostBounds.top) - hostBounds.top + 16),
          ] as [number, number],
          paddingBottomRight: [
            Math.max(28, hostBounds.right - (detailBounds?.left ?? hostBounds.right) + 18),
            Math.max(28, hostBounds.bottom - (legendBounds?.top ?? hostBounds.bottom) + 16),
          ] as [number, number],
        };
      };

      const fitCityBounds = (animate: boolean) => {
        activeMapArea.current = null;
        map.fitBounds(boundaryBounds, {
          ...resolveMapPadding(),
          animate,
        });
      };

      const focusAreaBounds = (area: AreaCode, animate = true) => {
        const bounds = areaBounds.current[area];
        if (!bounds) return;
        activeMapArea.current = area;
        map.fitBounds(bounds, {
          ...resolveMapPadding(),
          animate,
          maxZoom: 13,
        });
      };

      fitCityToViewport.current = fitCityBounds;
      focusAreaInViewport.current = (area) => focusAreaBounds(area);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · 행정동 경계: <a href="https://sgis.kostat.go.kr/">SGIS</a>',
        maxZoom: 19,
        noWrap: true,
      }).addTo(map);
      L.control.zoom({ position: "bottomright" }).addTo(map);

      // 사각형 마스크를 그리지 않고 지도 컨테이너 자체를 천안시 실제 경계로
      // 잘라서, 천안시 밖의 타일과 마스크 외곽선이 함께 보이지 않게 합니다.
      const cityMaskPane = map.createPane("city-mask");
      cityMaskPane.style.zIndex = "300";
      const cityMaskRenderer = L.canvas({
        pane: "city-mask",
        padding: 0,
      });
      const getViewportRing = (): import("leaflet").LatLngExpression[] => {
        const bounds = map.getBounds();
        return [
          [bounds.getSouth(), bounds.getWest()],
          [bounds.getSouth(), bounds.getEast()],
          [bounds.getNorth(), bounds.getEast()],
          [bounds.getNorth(), bounds.getWest()],
        ];
      };
      const cityMask = L.polygon([getViewportRing(), boundaryLatLngs], {
        pane: "city-mask",
        renderer: cityMaskRenderer,
        stroke: false,
        color: "transparent",
        weight: 0,
        opacity: 0,
        fill: true,
        fillColor: "#e8edf3",
        fillOpacity: 1,
        fillRule: "evenodd",
        interactive: false,
      }).addTo(map);
      const updateCityMask = () => {
        cityMask.setLatLngs([getViewportRing(), boundaryLatLngs]);
      };
      map.on("moveend zoomend resize", updateCityMask);

      const cityBoundaryPane = map.createPane("city-boundary");
      cityBoundaryPane.style.zIndex = "450";

      const labelMarkers: Array<{
        marker: import("leaflet").Marker;
        baseLatLng: import("leaflet").LatLng;
        polygonBounds: import("leaflet").LatLngBounds;
        name: string;
        isPilot: boolean;
        isPriority: boolean;
        width: number;
        height: number;
      }> = [];
      const labelOffsets = [
        [0, 0],
      ] as const;
      const getLabelDimensions = (entry: (typeof labelMarkers)[number]) => {
        const northWest = map.latLngToContainerPoint(entry.polygonBounds.getNorthWest());
        const southEast = map.latLngToContainerPoint(entry.polygonBounds.getSouthEast());
        const polygonWidth = Math.abs(southEast.x - northWest.x);
        const polygonHeight = Math.abs(southEast.y - northWest.y);
        const preferredFont = entry.isPilot || entry.isPriority ? 12 : 10;
        const minimumFont = entry.isPilot || entry.isPriority ? 8.5 : 7.5;
        const zoomScale = Math.min(1.12, Math.max(0.78, 0.78 + (map.getZoom() - 10) * 0.12));
        const horizontalPadding = map.getZoom() < 10.75 ? 3 : 4;
        const maximumTextWidth = Math.max(38, polygonWidth * 0.72);
        const polygonFontLimit = (maximumTextWidth - horizontalPadding * 2) / (entry.name.length * 1.05);
        const fontSize = Math.max(
          minimumFont,
          Math.min(preferredFont * zoomScale, polygonFontLimit),
        );
        const width = Math.max(38, Math.ceil(entry.name.length * fontSize * 1.05 + horizontalPadding * 2));
        const height = Math.ceil(fontSize + (fontSize < 10 ? 8 : 10));

        return { fontSize, horizontalPadding, width, height, polygonWidth, polygonHeight };
      };
      const updateLabelPositions = () => {
        const occupied: Array<{ left: number; top: number; right: number; bottom: number }> = [];
        const ordered = [...labelMarkers].sort(
          (left, right) =>
            Number(right.isPilot) - Number(left.isPilot) ||
            Number(right.isPriority) - Number(left.isPriority),
        );

        for (const entry of ordered) {
          const dimensions = getLabelDimensions(entry);
          entry.width = dimensions.width;
          entry.height = dimensions.height;
          const labelElement = entry.marker.getElement();
          labelElement?.style.setProperty("--label-font-size", `${dimensions.fontSize.toFixed(1)}px`);
          labelElement?.style.setProperty("--label-padding-x", `${dimensions.horizontalPadding}px`);
          labelElement?.style.setProperty("--label-min-width", `${dimensions.width}px`);
          labelElement?.style.setProperty("--label-min-height", `${dimensions.height}px`);
          const fitsInsidePolygon =
            entry.width <= dimensions.polygonWidth * 0.82 &&
            entry.height <= dimensions.polygonHeight * 0.7;
          if (!fitsInsidePolygon) {
            labelElement?.classList.add("is-collapsed");
            entry.marker.setLatLng(entry.baseLatLng);
            continue;
          }
          const base = map.latLngToContainerPoint(entry.baseLatLng);
          const offsets = labelOffsets;
          const chosen = offsets.find(([offsetX, offsetY]) => {
            const left = base.x + offsetX - entry.width / 2;
            const top = base.y + offsetY - entry.height / 2;
            const right = left + entry.width;
            const bottom = top + entry.height;
            return !occupied.some(
              (rect) =>
                left < rect.right && right > rect.left && top < rect.bottom && bottom > rect.top,
            );
          });
          if (!chosen) {
            labelElement?.classList.add("is-collapsed");
            continue;
          }
          labelElement?.classList.remove("is-collapsed");
          const [offsetX, offsetY] = chosen;
          const left = base.x + offsetX - entry.width / 2;
          const top = base.y + offsetY - entry.height / 2;
          occupied.push({
            left,
            top,
            right: left + entry.width,
            bottom: top + entry.height,
          });
          entry.marker.setLatLng(map.containerPointToLatLng(L.point(base.x + offsetX, base.y + offsetY)));
        }
      };
      map.on("moveend zoomend resize", updateLabelPositions);

      L.geoJSON(
        CHEONAN_SUBDIVISIONS as GeoJSON.FeatureCollection<
          GeoJSON.Geometry,
          SubdivisionProperties
        >,
        {
          style: (feature) => areaStyle(feature?.properties.name ?? "", PILOT_REGION_CODES[0], "equal"),
          onEachFeature: (feature, layer) => {
            const { district, kind, name } = feature.properties;
            const path = layer as import("leaflet").Path;
            const polygon = layer as import("leaflet").Polygon;
            areaLayers.current[name] = path;

            const kindClass = kind === "읍" ? "is-eup" : kind === "면" ? "is-myeon" : "is-dong";
            const stopCount = TRANSPORT_DATA[REGION_CODE_BY_NAME[name] ?? ""]?.stops;
            path.bindTooltip(
              `<strong>${name}</strong><span>${district} · ${kind}${stopCount !== undefined ? ` · 정류장 ${stopCount}개` : ""}${isPilotArea(name) ? " · 시범 분석지역" : ""}</span>`,
              { className: "subdivision-tooltip", sticky: true },
            );

            const code = REGION_CODE_BY_NAME[name];
            if (code) {
              areaBounds.current[code] = polygon.getBounds();
              path.on("click", () => {
                setSelectedArea(code);
                setDetailsOpen(false);
                focusAreaBounds(code);
              });
            }

            // 31개 행정구역 이름을 모두 표시하되, 시범지역은 큰 라벨로 강조한다.
            const isPilot = isPilotArea(name);
            const areaCode = REGION_CODE_BY_NAME[name];
            const isPriority = areaCode ? getVulnerabilitySummary(areaCode, "equal").rank <= 10 : false;
            const polygonBounds = polygon.getBounds();
            // onEachFeature 시점에는 Leaflet 레이어가 아직 지도에 등록되기 전이라
            // getCenter()를 호출할 수 없다. 경계 중심은 이 시점에도 안정적으로 쓸 수 있다.
            const labelCenter = polygonBounds.getCenter();
            const label = L.marker(labelCenter, {
              interactive: false,
              icon: L.divIcon({
                className: `subdivision-name ${kindClass} ${isPilot ? "is-pilot" : isPriority ? "is-priority" : "is-all"}`,
                html: `<span>${name}</span>`,
                iconSize: [100, 34],
                iconAnchor: [50, 17],
              }),
            });
            label.addTo(map);
            labelMarkers.push({
              marker: label,
              baseLatLng: labelCenter,
              polygonBounds,
              name,
              isPilot,
              isPriority,
              width: 72,
              height: 28,
            });
          },
        },
      ).addTo(map);

      L.polygon(boundaryLatLngs, {
        pane: "city-boundary",
        color: "#1d4ed8",
        weight: 2.5,
        opacity: 0.9,
        fill: false,
        interactive: false,
      }).addTo(map);

      // 초기 화면과 '천안 전체보기'는 시범지역 일부가 아니라 천안시 전체 경계를
      // 기준으로 맞춘다. 특정 지역은 탭/경계 클릭 때만 해당 영역으로 이동한다.
      fitCityBounds(false);
      updateCityMask();
      updateLabelPositions();
      mapInstance.current = map;
      cityBounds.current = boundaryBounds;
      setMapReady(true);

      const handleResize = () => {
        map.invalidateSize({ pan: false, debounceMoveend: true });
        window.requestAnimationFrame(() => {
          const activeArea = activeMapArea.current;
          if (activeArea) {
            focusAreaBounds(activeArea, false);
            return;
          }
          fitCityBounds(false);
        });
      };
      window.addEventListener("resize", handleResize);

      return () => window.removeEventListener("resize", handleResize);
    }

    let removeResizeListener: (() => void) | undefined;
    void createMap().then((cleanup) => {
      removeResizeListener = cleanup;
    });
    return () => {
      disposed = true;
      removeResizeListener?.();
      cleanupMap();
    };
  }, []);

  useEffect(() => {
    for (const [name, layer] of Object.entries(areaLayers.current)) {
      layer.setStyle(areaStyle(name, selectedArea, weightingModel));
    }
  }, [selectedArea, weightingModel]);

  useEffect(() => {
    detailCard.current?.scrollTo({ top: 0, behavior: "auto" });
  }, [selectedArea]);

  useEffect(() => {
    mapFocusModeRef.current = mapFocusMode;
    const frame = window.requestAnimationFrame(() => {
      mapInstance.current?.invalidateSize({ pan: false, debounceMoveend: true });
      const activeArea = activeMapArea.current;
      if (activeArea) {
        focusAreaInViewport.current(activeArea);
        return;
      }
      fitCityToViewport.current(true);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [mapFocusMode]);

  const profile = AREA_PROFILES[selectedArea];
  const population = POPULATION_DATA[selectedArea];
  const transport = TRANSPORT_DATA[selectedArea];
  const transportMetrics = getTransportMetrics(selectedArea);
  const pilotAreas = ALL_REGION_CODES;
  const selectedModel = VULNERABILITY_MODELS[weightingModel];
  const getAreaVulnerability = (area: AreaCode) => getVulnerabilitySummary(area, weightingModel);
  const vulnerability = getAreaVulnerability(selectedArea);
  const rankedAreas = [...pilotAreas].sort(
    (left, right) => getAreaVulnerability(left).rank - getAreaVulnerability(right).rank,
  );
  const mapRankAreas = rankedAreas;
  const comparisonCandidates = comparisonScope === "top10" ? rankedAreas.slice(0, 10) : [...pilotAreas];
  const sortedComparisonAreas = [...comparisonCandidates].sort((left, right) => {
    if (comparisonSort === "name") return pilotName(left).localeCompare(pilotName(right), "ko");
    if (comparisonSort === "elderly") {
      return POPULATION_DATA[right].elderly - POPULATION_DATA[left].elderly || pilotName(left).localeCompare(pilotName(right), "ko");
    }
    return getAreaVulnerability(left).rank - getAreaVulnerability(right).rank;
  });
  const comparisonAreas = [selectedArea, ...sortedComparisonAreas.filter((area) => area !== selectedArea)];
  const comparisonAreaSplit = Math.ceil(comparisonAreas.length / 2);
  const comparisonAreaColumns = [comparisonAreas.slice(0, comparisonAreaSplit), comparisonAreas.slice(comparisonAreaSplit)];
  const maxElderly = Math.max(...pilotAreas.map((area) => POPULATION_DATA[area].elderly));
  const maxAgeBand = Math.max(0, ...pilotAreas.flatMap((area) => POPULATION_DATA[area].ageBands ?? []));
  const maxDisabled = Math.max(...pilotAreas.map((area) => POPULATION_DATA[area].disabled ?? 0));

  function resetMap() {
    fitCityToViewport.current(true);
  }

  return (
    <main className={`map-shell ${mapFocusMode ? "is-map-focused" : ""}`}>
      <div ref={mapElement} className="map" aria-label="천안시 읍·면·동 고령·교통 취약 지도" />
      {!mapReady && <div className="map-loading">시범지역 지도를 불러오는 중입니다</div>}

      <header className="map-header">
        <div>
            <p className="eyebrow">천안시 순회진료 의사결정 지원 · 31개 행정구역 비교</p>
          <h1>천안시 읍면동 지도</h1>
        </div>
        <nav className="map-actions" aria-label="지도 보기 제어">
          <button
            type="button"
            className="map-focus-button"
            aria-pressed={mapFocusMode}
            onClick={() => setMapFocusMode((value) => !value)}
          >
            {mapFocusMode ? "패널 보기" : "지도 집중 모드"}
          </button>
          <button type="button" className="reset-button" onClick={resetMap}>
            전체 지도 보기
          </button>
        </nav>
      </header>

      <aside ref={leftRail} className="map-left-rail" aria-label="지도 제어 패널">

      <nav className="pilot-tabs" aria-label="행정구역 선택">
        <label className="region-select-label" htmlFor="region-selector">
          <span>행정구역 선택</span>
          <select
            id="region-selector"
            value={selectedArea}
            onChange={(event) => {
              const code = event.target.value;
              setSelectedArea(code);
              setDetailsOpen(false);
              focusAreaInViewport.current(code);
            }}
          >
            {pilotAreas.map((area) => {
              const region = REGION_BY_NAME[pilotName(area)];
              const areaVulnerability = getAreaVulnerability(area);
              return <option key={area} value={area}>{pilotName(area)} · {region?.district} · {areaVulnerability.grade}등급</option>;
            })}
          </select>
          <small>지도 경계를 클릭해도 지역을 선택할 수 있습니다.</small>
        </label>
      </nav>

      <section className="map-rank-card" aria-label="취약지역 순위">
        <div className="map-rank-head">
          <div>
            <span className="eyebrow">취약지역 우선순위</span>
            <strong>전체 31개 지역</strong>
          </div>
          <span className="rank-model-badge">{selectedModel.label}</span>
        </div>
        <div className="map-rank-legend" aria-label="취약순위 색상 범례">
          <span><i className="rank-legend-swatch is-critical" />1–5위</span>
          <span><i className="rank-legend-swatch is-high" />6–10위</span>
          <span><i className="rank-legend-swatch is-watch" />11–20위</span>
          <span><i className="rank-legend-swatch is-safe" />21–31위</span>
        </div>
        <div className="map-rank-list">
          {mapRankAreas.map((area) => {
            const summary = getAreaVulnerability(area);
            const urgency = summary.rank <= 5 ? "critical" : summary.rank <= 10 ? "high" : "watch";
            return (
              <button
                type="button"
                className={`map-rank-row ${selectedArea === area ? "is-selected" : ""}`}
                key={area}
                onClick={() => {
                  setSelectedArea(area);
                  setDetailsOpen(false);
                  focusAreaInViewport.current(area);
                }}
              >
                <span className={`rank-dot rank-${urgency}`}>{summary.rank}</span>
                <span className="map-rank-name">{pilotName(area)}</span>
                <span className={`map-rank-grade is-${urgency}`}>{summary.grade}등급</span>
                <b>{summary.overallScore.toFixed(1)}</b>
              </button>
            );
          })}
        </div>
        <small>점수순 정렬 · 지도 색상과 연동</small>
      </section>

      </aside>

      <aside ref={detailCard} className={`area-card is-${profile.tone}`}>
        <div className="area-card-head">
          <div>
            <span className="status-badge">{vulnerability.grade}등급 · {vulnerability.level}</span>
            <h2>{pilotName(selectedArea)}</h2>
            <p>인구 2026년 6월 · 면적 {AREA_DATA[selectedArea].source}</p>
          </div>
          <div className="access-score">
            <strong>{population.elderly.toLocaleString()}</strong>
            <span>65세 이상 인구</span>
          </div>
        </div>

        <section className="vulnerability-panel" aria-labelledby="vulnerability-title">
          <div className="vulnerability-head">
            <div>
              <span className="module-kicker">고령·교통 취약성 모듈</span>
              <strong id="vulnerability-title">고령 수요·교통 접근성 요약</strong>
            </div>
            <div className={`module-score is-${weightingModel}`} aria-label={`${pilotName(selectedArea)} 종합 취약점수`}>
              {vulnerability.overallScore.toFixed(1)}
            </div>
          </div>
          <div className="model-selector">
            <label htmlFor="weighting-model">종합점수 기준</label>
            <select
              id="weighting-model"
              value={weightingModel}
              onChange={(event) => setWeightingModel(event.target.value as WeightingModel)}
            >
              {(Object.keys(VULNERABILITY_MODELS) as WeightingModel[]).map((modelKey) => (
                <option key={modelKey} value={modelKey}>{VULNERABILITY_MODELS[modelKey].label}</option>
              ))}
            </select>
            <span>{selectedModel.description}</span>
          </div>
          <div className="vulnerability-grid">
            <div>
              <span>65세 이상 비율</span>
              <strong>{population.elderlyRate.toFixed(1)}%</strong>
              <small>전체 인구 대비</small>
            </div>
            <div>
              <span>정류장</span>
              <strong>{transport.stops}개</strong>
              <small>행정구역 내 집계</small>
            </div>
            <div>
              <span>고령인구 1,000명당</span>
              <strong>{transportMetrics.supply.toFixed(1)}개</strong>
              <small>정류장 공급률</small>
            </div>
            <div>
              <span>고령인구 밀도</span>
              <strong>{getElderlyDensity(selectedArea).toFixed(1)}명/㎢</strong>
              <small>65세 이상 인구 ÷ 면적</small>
            </div>
          </div>
          <p className="module-note">
            면적 {AREA_DATA[selectedArea].areaKm2.toFixed(2)}㎢ · 정류장 자료 기준일 {transport.sourceDate} · {transport.method}
          </p>
          <p className="module-note">
            종합 {vulnerability.overallScore.toFixed(1)}점 · 인구취약 {vulnerability.populationScore.toFixed(1)}점 · 의료취약 {vulnerability.medicalScore.toFixed(1)}점 · 교통취약 {vulnerability.transportScore.toFixed(1)}점 · {vulnerability.coreArea} 중심
          </p>
          <p className="module-note">
            현재 기준: {selectedModel.label} · 인구 {formatWeight(selectedModel.weights.population)} · 의료 {formatWeight(selectedModel.weights.medical)} · 교통 {formatWeight(selectedModel.weights.transport)} · 전체 31개 지역 순위
          </p>
          <p className="module-note"><b>정책추천:</b> {vulnerability.policyRecommendation}</p>
          <details className="calculation-details">
            <summary>도출 과정 보기</summary>
            <p>고령인구 밀도 = 65세 이상 인구 {population.elderly.toLocaleString()}명 ÷ 면적 {AREA_DATA[selectedArea].areaKm2.toFixed(2)}㎢ = <b>{getElderlyDensity(selectedArea).toFixed(1)}명/㎢</b></p>
            <p>정류장 공급률 = 정류장 {transport.stops}개 ÷ 65세 이상 인구 {population.elderly.toLocaleString()}명 × 1,000 = <b>{transportMetrics.supply.toFixed(1)}개/1,000명</b></p>
            <p>면적 출처: {AREA_DATA[selectedArea].source}</p>
          </details>
        </section>

        <section className="resource-summary" aria-label="의료자원과 사회취약 현황">
          <div className="resource-summary-head">
            <strong>의료자원·사회취약 현황</strong>
            <span>의료기관·약국 2024 · 독거노인 2019 · 장애인 2023</span>
          </div>
          <div className="stat-grid">
            <div><span>의료기관</span><strong>{formatCount(profile.medicalInstitutions)}{profile.medicalInstitutions !== null ? "곳" : ""}</strong></div>
            <div><span>공공 보건기관</span><strong>{formatCount(profile.publicInstitutions)}{profile.publicInstitutions !== null ? "곳" : ""}</strong></div>
            <div><span>약국</span><strong>{formatCount(profile.pharmacy ?? null)}{profile.pharmacy !== null && profile.pharmacy !== undefined ? "곳" : ""}</strong></div>
            <div><span>독거노인</span><strong>{population.livingAlone === null ? "자료 없음" : `${population.livingAlone.toLocaleString()}명`}</strong></div>
            <div><span>등록장애인</span><strong>{population.disabled === null ? "자료 없음" : `${population.disabled.toLocaleString()}명`}</strong></div>
          </div>
        </section>

        <button
          type="button"
          className="detail-toggle"
          onClick={() => setDetailsOpen((open) => !open)}
        >
          {detailsOpen ? "시설 세부정보 닫기" : "확인된 시설 세부정보 보기"}
        </button>

        <button
          type="button"
          className="comparison-button"
          onClick={() => setComparisonOpen(true)}
        >
          고령·사회취약 현황 비교
        </button>

        {detailsOpen && (
          <div className="facility-list">
            {profile.verified.length === 0 && (
              <p className="empty-facility-note">2024년 통계 집계는 반영했지만, 현재 연결된 공개 명단에서 이 지역의 개별 시설명·전화번호를 확인하지 못했습니다.</p>
            )}
            {profile.verified.map((facility) => (
              <article key={facility.name}>
                <span>{facility.type}</span>
                <strong>{facility.name}</strong>
                <p>{facility.address}</p>
                <a
                  href={`https://map.kakao.com/link/search/${encodeURIComponent(facility.name)}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {facility.phone} · 지도에서 보기
                </a>
              </article>
            ))}
            <small>
              2024년 KOSIS 집계와 최신 공개 명단이 다른 경우, 실제 시설 탐색에는
              2026년 공개 명단을 우선 적용했습니다.
            </small>
          </div>
        )}
      </aside>

      {comparisonOpen && (
        <div className="comparison-backdrop" role="presentation">
          <section
            className="comparison-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="comparison-title"
          >
            <div className="comparison-scroll-content">
            <header className="comparison-head">
              <div>
                <p className="eyebrow">원자료·가중치 모형 비교</p>
                <h2 id="comparison-title">천안시 31개 행정구역 고령·사회취약 현황</h2>
                <p>
                  기본은 동일가중치이며, 위의 종합점수 기준을 바꾸면 아래 취약지역 점수도
                  같은 모형 기준으로 비교됩니다. 전체 연령을 모두 나열하지 않고 순회진료와
                  직접 관련된 65세 이상 구간을 중심으로 표시합니다.
                </p>
              </div>
              <button type="button" onClick={() => setComparisonOpen(false)}>
                닫기
              </button>
            </header>

            <section className="metric-focus-card" aria-labelledby="comparison-metric-title">
              <div className="metric-focus-head">
                <div>
                  <span className="metric-focus-kicker">선택 항목 집중 비교</span>
                  <strong id="comparison-metric-title">{COMPARISON_METRICS[comparisonMetric]}</strong>
                </div>
                <div className="metric-focus-tools">
                  <div className="metric-selector">
                    <label htmlFor="comparison-metric">비교 항목</label>
                    <select
                      id="comparison-metric"
                      value={comparisonMetric}
                      onChange={(event) => setComparisonMetric(event.target.value as ComparisonMetric)}
                    >
                      {(Object.keys(COMPARISON_METRICS) as ComparisonMetric[]).map((metric) => (
                        <option key={metric} value={metric}>{COMPARISON_METRICS[metric]}</option>
                      ))}
                    </select>
                  </div>
                  <div className="metric-selector">
                    <label htmlFor="comparison-scope">표시 범위</label>
                    <select
                      id="comparison-scope"
                      value={comparisonScope}
                      onChange={(event) => setComparisonScope(event.target.value as ComparisonScope)}
                    >
                      <option value="all">전체 31개</option>
                      <option value="top10">취약 상위 10개 + 선택지역</option>
                    </select>
                  </div>
                  <div className="metric-selector">
                    <label htmlFor="comparison-sort">정렬 기준</label>
                    <select
                      id="comparison-sort"
                      value={comparisonSort}
                      onChange={(event) => setComparisonSort(event.target.value as ComparisonSort)}
                    >
                      {(Object.keys(COMPARISON_SORT_LABELS) as ComparisonSort[]).map((sort) => (
                        <option key={sort} value={sort}>{COMPARISON_SORT_LABELS[sort]}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {comparisonMetric === "elderly-count" && (
                <div className="focus-bar-chart">
                  <div className="focus-chart-axis"><span>0명</span><span>{maxElderly.toLocaleString()}명</span></div>
                  {comparisonAreas.map((area) => {
                    const value = POPULATION_DATA[area].elderly;
                    return (
                      <div className="focus-bar-row" key={area}>
                        <strong>{pilotName(area)}</strong>
                        <div className="focus-bar-track"><i style={{ width: `${(value / maxElderly) * 100}%` }} /></div>
                        <b>{value.toLocaleString()}명</b>
                      </div>
                    );
                  })}
                </div>
              )}

              {comparisonMetric === "elderly-rate" && (
                <div className="focus-bar-chart">
                  <div className="focus-chart-axis"><span>0%</span><span>50%</span></div>
                  {comparisonAreas.map((area) => {
                    const value = POPULATION_DATA[area].elderlyRate;
                    return (
                      <div className="focus-bar-row" key={area}>
                        <strong>{pilotName(area)}</strong>
                        <div className="focus-bar-track is-rate"><i style={{ width: `${(value / 50) * 100}%` }} /></div>
                        <b>{value.toFixed(1)}%</b>
                      </div>
                    );
                  })}
                </div>
              )}

              {comparisonMetric === "age-band" && (
                <div className="focus-band-chart">
                  {AGE_BAND_LABELS.map((label, index) => (
                    <div className="focus-band-group" key={label}>
                      <strong>{label}</strong>
                      {comparisonAreas.map((area) => {
                        const value = POPULATION_DATA[area].ageBands?.[index] ?? null;
                        return (
                          <div className="focus-band-row" key={area}>
                            <span>{pilotName(area)}</span>
                            {value === null ? <b>자료 없음</b> : <><div><i style={{ width: `${maxAgeBand ? (value / maxAgeBand) * 100 : 0}%` }} /></div><b>{value.toLocaleString()}</b></>}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              )}

              {comparisonMetric === "living-alone" && (
                <div className="focus-stack-chart">
                  {comparisonAreas.map((area) => {
                    const item = POPULATION_DATA[area];
                    const available = item.livingAlone !== null && item.basicLiving !== null && item.lowIncome !== null;
                    const total = item.livingAlone ?? 0;
                    const general = available ? total - (item.basicLiving ?? 0) - (item.lowIncome ?? 0) : 0;
                    return (
                      <div className="focus-stack-row" key={area}>
                        <strong>{pilotName(area)}</strong>
                        {available ? (
                          <>
                            <div className="focus-stack-track" aria-label={`${pilotName(area)} 독거노인 구성`}>
                              <i className="is-basic" style={{ width: `${((item.basicLiving ?? 0) / total) * 100}%` }} />
                              <i className="is-low" style={{ width: `${((item.lowIncome ?? 0) / total) * 100}%` }} />
                              <i className="is-general" style={{ width: `${(general / total) * 100}%` }} />
                            </div>
                            <b>총 {total.toLocaleString()}명</b>
                          </>
                        ) : <b>자료 없음</b>}
                      </div>
                    );
                  })}
                  <div className="focus-stack-legend"><span><i className="is-basic" />기초생활보장</span><span><i className="is-low" />저소득</span><span><i className="is-general" />일반</span></div>
                </div>
              )}

              {comparisonMetric === "disabled" && (
                <div className="focus-bar-chart">
                  <div className="focus-chart-axis"><span>0명</span><span>{maxDisabled.toLocaleString()}명</span></div>
                  {comparisonAreas.map((area) => {
                    const item = POPULATION_DATA[area];
                    const value = item.disabled ?? 0;
                    const rate = item.disabled === null ? null : (item.disabled / item.total) * 100;
                    return (
                      <div className="focus-bar-row" key={area}>
                        <strong>{pilotName(area)}</strong>
                        <div className="focus-bar-track is-disabled"><i style={{ width: `${(value / maxDisabled) * 100}%` }} /></div>
                        <b>{item.disabled === null ? "자료 없음" : `${value.toLocaleString()}명 · ${rate?.toFixed(1)}%`}</b>
                      </div>
                    );
                  })}
                </div>
              )}

              <p className="metric-focus-note">
                선택 항목의 막대 길이는 지역 간 비교를 위한 상대 크기입니다. 현재 {comparisonAreas.length}개 지역 표시 · {comparisonScope === "top10" ? "취약 상위 10개 + 선택 지역" : "전체 31개 지역"} · {COMPARISON_SORT_LABELS[comparisonSort]} · 선택 지역은 항상 맨 위에 고정됩니다.
              </p>
            </section>

            <div className="population-summary">
              {comparisonAreas.map((area) => {
                const item = POPULATION_DATA[area];
                return (
                  <article key={area} className={selectedArea === area ? "is-selected" : ""}>
                    <strong>{pilotName(area)}</strong>
                    <div><span>전체 인구</span><b>{item.total.toLocaleString()}명</b></div>
                    <div><span>65세 이상</span><b>{item.elderly.toLocaleString()}명</b></div>
                    <div><span>고령인구 비율</span><b>{item.elderlyRate.toFixed(1)}%</b></div>
                  </article>
                );
              })}
            </div>

            <article className="chart-card is-wide age-composition-card">
              <div className="chart-title">
                <strong>65세 이상 연령구성</strong>
                <span>선택지역 도넛 · 자료가 있는 지역 비교</span>
              </div>
              <div className="age-composition-layout">
                {(() => {
                  const values = population.ageComposition;
                  if (!values) return <div className="age-data-empty">선택 지역의 연령구성 원자료가 아직 없습니다.</div>;
                  const elderlyTotal = values.reduce((sum, value) => sum + value, 0);
                  const first = (values[0] / elderlyTotal) * 100;
                  const second = first + (values[1] / elderlyTotal) * 100;
                  return (
                    <div className="age-donut-wrap">
                      <div
                        className="age-donut"
                        style={{ background: `conic-gradient(${AGE_COMPOSITION_COLORS[0]} 0 ${first}%, ${AGE_COMPOSITION_COLORS[1]} ${first}% ${second}%, ${AGE_COMPOSITION_COLORS[2]} ${second}% 100%)` }}
                        aria-label={`${pilotName(selectedArea)} 65세 이상 연령구성`}
                      >
                        <span><b>{elderlyTotal.toLocaleString()}</b><small>65세 이상</small></span>
                      </div>
                      <div className="age-legend">
                        {values.map((value, index) => (
                          <span key={AGE_COMPOSITION_LABELS[index]}>
                            <i style={{ background: AGE_COMPOSITION_COLORS[index] }} />
                            {AGE_COMPOSITION_LABELS[index]} {value.toLocaleString()}명 ({((value / elderlyTotal) * 100).toFixed(1)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })()}
                <div className="age-stack-compare">
                  {comparisonAreas.map((area) => {
                    const values = POPULATION_DATA[area].ageComposition;
                    if (!values) return <div className="age-stack-row" key={area}><strong>{pilotName(area)}</strong><b>자료 없음</b></div>;
                    const total = values.reduce((sum, value) => sum + value, 0);
                    let offset = 0;
                    return (
                      <div className="age-stack-row" key={area}>
                        <strong>{pilotName(area)}</strong>
                        <div className="age-stack-track" aria-label={`${pilotName(area)} 연령구성 비교`}>
                          {values.map((value, index) => {
                            const width = (value / total) * 100;
                            const part = <i key={AGE_COMPOSITION_LABELS[index]} style={{ left: `${offset}%`, width: `${width}%`, background: AGE_COMPOSITION_COLORS[index] }} />;
                            offset += width;
                            return part;
                          })}
                        </div>
                        <b>{total.toLocaleString()}명</b>
                      </div>
                    );
                  })}
                  <small>구간: 65~74세 · 75~84세 · 85세 이상</small>
                </div>
              </div>
              <p className="source-note">연령구성은 2026년 6월 주민등록 연령별 인구 원자료의 5세 구간을 합산했습니다. 65~74세=65~69세+70~74세, 75~84세=75~79세+80~84세, 85세 이상=85세 이상 전체입니다.</p>
            </article>

            <div className="comparison-grid">
              <article className="chart-card">
                <div className="chart-title">
                  <strong>65세 이상 인구수</strong>
                  <span>동일 축 · 현재 비교값 중 최댓값 {maxElderly.toLocaleString()}명</span>
                </div>
                {comparisonAreas.map((area) => {
                  const item = POPULATION_DATA[area];
                  return (
                    <div className="bar-row" key={area}>
                      <span>{pilotName(area)}</span>
                      <div className="bar-track">
                        <i style={{ width: `${(item.elderly / maxElderly) * 100}%` }} />
                      </div>
                      <b>{item.elderly.toLocaleString()}명</b>
                    </div>
                  );
                })}
              </article>

              <article className="chart-card">
                <div className="chart-title">
                  <strong>전체 인구 대비 65세 이상 비율</strong>
                  <span>0–50% 동일 축</span>
                </div>
                {comparisonAreas.map((area) => {
                  const item = POPULATION_DATA[area];
                  return (
                    <div className="bar-row" key={area}>
                      <span>{pilotName(area)}</span>
                      <div className="bar-track is-rate">
                        <i style={{ width: `${(item.elderlyRate / 50) * 100}%` }} />
                      </div>
                      <b>{item.elderlyRate.toFixed(1)}%</b>
                    </div>
                  );
                })}
              </article>

              <article className="chart-card is-wide vulnerability-chart">
                <div className="chart-title vulnerability-chart-title">
                  <div>
                    <strong>취약지역 종합점수 비교</strong>
                    <span>{selectedModel.label} 기준 · 인구 {formatWeight(selectedModel.weights.population)} · 의료 {formatWeight(selectedModel.weights.medical)} · 교통 {formatWeight(selectedModel.weights.transport)}</span>
                  </div>
                  <div className="comparison-model-selector">
                    <label htmlFor="comparison-weighting-model">비교 기준</label>
                    <select
                      id="comparison-weighting-model"
                      value={weightingModel}
                      onChange={(event) => setWeightingModel(event.target.value as WeightingModel)}
                    >
                      {(Object.keys(VULNERABILITY_MODELS) as WeightingModel[]).map((modelKey) => (
                        <option key={modelKey} value={modelKey}>{VULNERABILITY_MODELS[modelKey].label}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className={`vulnerability-graph is-${weightingModel}`} aria-label={`${selectedModel.label} 기준 천안시 31개 행정구역 종합점수 그래프`}>
                  <div className="vulnerability-graph-columns">
                    {comparisonAreaColumns.map((column, columnIndex) => (
                      <div className="vulnerability-column" key={columnIndex}>
                        <div className="vulnerability-graph-scale" aria-hidden="true">
                          <span />
                          <div><span>0</span><span>25</span><span>50</span><span>75</span><span>100점</span></div>
                        </div>
                        {column.map((area) => {
                          const summary = getAreaVulnerability(area);
                          const barWidth = Math.min(100, Math.max(0, summary.overallScore));
                          return (
                            <div className="vulnerability-graph-row" key={area}>
                              <div className="vulnerability-graph-label">
                                <strong>{summary.rank}. {pilotName(area)}</strong>
                                <span>{summary.grade}등급</span>
                              </div>
                              <div className="vulnerability-graph-track">
                                <i style={{ width: `${barWidth}%` }} />
                                <b>{summary.overallScore.toFixed(1)}점</b>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                  <small>막대가 길수록 해당 모형에서 취약점수가 높습니다. 등급·순위는 천안시 전체 31개 지역 기준입니다.</small>
                </div>
                <div className="vulnerability-detail-label">영역별 상세 점수</div>
                {comparisonAreas.map((area) => {
                  const summary = getAreaVulnerability(area);
                  return (
                    <div className="vulnerability-row" key={area}>
                      <strong>{pilotName(area)}</strong>
                      <div><span>등급·순위</span><b>{summary.grade} · {summary.rank}위</b></div>
                      <div><span>종합점수</span><b>{summary.overallScore.toFixed(1)}점</b></div>
                      <div><span>인구취약</span><b>{summary.populationScore.toFixed(1)}점</b></div>
                      <div><span>의료취약</span><b>{summary.medicalScore.toFixed(1)}점</b></div>
                      <div><span>교통취약</span><b>{summary.transportScore.toFixed(1)}점</b></div>
                    </div>
                  );
                })}
                <p className="source-note">
                  출처: 동일가중치_CRITIC_가중치_비교분석.xlsx의 31개 지역별결과. 천안시 전체 행정구역을 표시하며, 점수는 개인 의료진단이 아닌 행정지원 우선순위 비교지표입니다.
                </p>
              </article>

              <article className="chart-card is-wide">
                <div className="chart-title">
                  <strong>65세 이상 연령구간</strong>
                  <span>모든 막대 동일 축 · 현재 비교값 중 최댓값 {maxAgeBand.toLocaleString()}명</span>
                </div>
                {AGE_BAND_LABELS.map((label, index) => (
                  <div className="band-group" key={label}>
                    <span>{label}</span>
                    <div className="band-columns">
                      {comparisonAreaColumns.map((column, columnIndex) => (
                        <div className="band-column" key={columnIndex}>
                          {column.map((area) => (
                            <div className="mini-bar" key={area}>
                              <small>{pilotName(area)}</small>
                              {(() => {
                                const value = POPULATION_DATA[area].ageBands?.[index] ?? null;
                                return value === null ? <b>자료 없음</b> : <><div><i style={{ width: `${maxAgeBand ? (value / maxAgeBand) * 100 : 0}%` }} /></div><b>{value.toLocaleString()}</b></>;
                              })()}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </article>

              <article className="chart-card is-wide">
                <div className="chart-title">
                  <strong>독거노인 구성</strong>
                  <span>2019년 · 하위 항목 중복 없음</span>
                </div>
                {comparisonAreas.map((area) => {
                  const item = POPULATION_DATA[area];
                  const available = item.livingAlone !== null && item.basicLiving !== null && item.lowIncome !== null;
                  const general = available ? item.livingAlone - item.basicLiving - item.lowIncome : 0;
                  return (
                    <div className="stacked-row" key={area}>
                      <strong>{pilotName(area)}</strong>
                      {available ? <>
                        <div className="stacked-track" aria-label={`${pilotName(area)} 독거노인 구성`}>
                          <i className="is-basic" style={{ width: `${(item.basicLiving / item.livingAlone) * 100}%` }} />
                          <i className="is-low" style={{ width: `${(item.lowIncome / item.livingAlone) * 100}%` }} />
                          <i className="is-general" style={{ width: `${(general / item.livingAlone) * 100}%` }} />
                        </div>
                        <b>총 {item.livingAlone.toLocaleString()}명</b>
                      </> : <b>자료 없음</b>}
                    </div>
                  );
                })}
                <div className="stacked-legend">
                  <span><i className="is-basic" />기초생활보장</span>
                  <span><i className="is-low" />저소득</span>
                  <span><i className="is-general" />일반</span>
                </div>
              </article>

              <article className="chart-card is-wide disabled-chart">
                <div className="chart-title">
                  <strong>등록장애인 수와 전체 인구 대비 비율</strong>
                  <span>2023년 · 전체 연령</span>
                </div>
                {comparisonAreas.map((area) => {
                  const item = POPULATION_DATA[area];
                  const rate = item.disabled === null ? null : (item.disabled / item.total) * 100;
                  return (
                    <div className="bar-row" key={area}>
                        <span>{pilotName(area)}</span>
                      <div className="bar-track">
                        <i style={{ width: `${item.disabled === null ? 0 : (item.disabled / maxDisabled) * 100}%` }} />
                      </div>
                      <b>{item.disabled === null ? "자료 없음" : `${item.disabled.toLocaleString()}명 · ${rate?.toFixed(1)}%`}</b>
                    </div>
                  );
                })}
              </article>
            </div>

            <div className="data-table-wrap">
              <table>
                <thead>
                  <tr><th>지표</th><th>기준</th>{comparisonAreas.map((area) => <th key={area}>{pilotName(area)}</th>)}<th>해석 주의</th></tr>
                </thead>
                <tbody>
                  <tr><th>전체 인구</th><td>2026.6</td>{comparisonAreas.map((area) => <td key={area}>{POPULATION_DATA[area].total.toLocaleString()}명</td>)}<td>외국인 제외</td></tr>
                  <tr><th>65세 이상 인구</th><td>2026.6</td>{comparisonAreas.map((area) => <td key={area}>{POPULATION_DATA[area].elderly.toLocaleString()}명</td>)}<td>주민등록인구</td></tr>
                  <tr><th>65세 이상 비율</th><td>2026.6</td>{comparisonAreas.map((area) => <td key={area}>{POPULATION_DATA[area].elderlyRate.toFixed(1)}%</td>)}<td>전체 인구 대비</td></tr>
                  <tr><th>약국</th><td>2024</td>{comparisonAreas.map((area) => <td key={area}>{formatCount(AREA_PROFILES[area].pharmacy ?? null)}{AREA_PROFILES[area].pharmacy !== null && AREA_PROFILES[area].pharmacy !== undefined ? "곳" : ""}</td>)}<td>읍·면·동별 약국 수</td></tr>
                  <tr><th>독거노인</th><td>2019</td>{comparisonAreas.map((area) => <td key={area}>{formatCount(POPULATION_DATA[area].livingAlone)}{POPULATION_DATA[area].livingAlone !== null ? "명" : ""}</td>)}<td>최신 연도 차이 주의</td></tr>
                  <tr><th>기초생활보장 독거노인</th><td>2019</td>{comparisonAreas.map((area) => <td key={area}>{formatCount(POPULATION_DATA[area].basicLiving)}{POPULATION_DATA[area].basicLiving !== null ? "명" : ""}</td>)}<td>독거노인 중 하위 항목</td></tr>
                  <tr><th>저소득 독거노인</th><td>2019</td>{comparisonAreas.map((area) => <td key={area}>{formatCount(POPULATION_DATA[area].lowIncome)}{POPULATION_DATA[area].lowIncome !== null ? "명" : ""}</td>)}<td>독거노인 중 하위 항목</td></tr>
                  <tr><th>등록장애인</th><td>2023</td>{comparisonAreas.map((area) => <td key={area}>{formatCount(POPULATION_DATA[area].disabled)}{POPULATION_DATA[area].disabled !== null ? "명" : ""}</td>)}<td>전체 연령</td></tr>
                </tbody>
              </table>
            </div>

            <footer className="source-note">
              출처: KOSIS 행정구역(읍면동)별/5세별 주민등록인구(2026.6), 독거노인 현황(2019),
              의약품 등 판매업소 현황(2024), 장애인 등록현황(2023), 국토교통부 전국 버스정류장 위치정보(2025.10.31).
              읍면동별 통계는 항목별 기준연도가 다르므로 화면의 기준연도를 함께 확인합니다.
              교통 지표는 행정구역별 공급 비교용 보조지표이며, 최종 취약순위와 기준자료가 다를 수 있습니다.
            </footer>
            </div>
            <footer className="comparison-action-footer">
            <button
              type="button"
              className="comparison-close-bottom"
              onClick={() => setComparisonOpen(false)}
            >
              닫기
            </button>
            </footer>
          </section>
        </div>
      )}

      <div ref={mapLegend} className="map-legend">
        <strong>읍·면·동 구분</strong>
        <div className="legend-items">
          <span><i className="legend-swatch is-eup" />읍</span>
          <span><i className="legend-swatch is-myeon" />면</span>
          <span><i className="legend-swatch is-dong" />동</span>
        </div>
        <small>구·읍·면·동은 행정구역 속성값으로 분리하며, 기존 시범 4개 지역은 비교지역으로 테두리를 강조합니다.</small>
      </div>
    </main>
  );
}
