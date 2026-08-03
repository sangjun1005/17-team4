import CHEONAN_SUBDIVISIONS from "./cheonan-subdivisions.json";

export type RegionKind = "읍" | "면" | "동";

export type RegionProperties = {
  code: string;
  name: string;
  district: "동남구" | "서북구";
  kind: RegionKind;
};

export const REGIONS = CHEONAN_SUBDIVISIONS.features.map(
  (feature) => feature.properties as RegionProperties,
);

export const REGION_BY_CODE: Record<string, RegionProperties> = Object.fromEntries(
  REGIONS.map((region) => [region.code, region]),
);

export const REGION_CODE_BY_NAME: Record<string, string> = Object.fromEntries(
  REGIONS.map((region) => [region.name, region.code]),
);

export const REGION_BY_NAME: Record<string, RegionProperties> = Object.fromEntries(
  REGIONS.map((region) => [region.name, region]),
);

export const ALL_REGION_CODES = REGIONS.map((region) => region.code);
export type RegionCode = string;

// 상세 대시보드가 먼저 연결된 4개 시범지역.
// 나머지 지역 자료가 추가되면 이 배열에 코드를 넣는 대신, 지역 데이터만 보강하면 된다.
export const PILOT_REGION_CODES = [
  "4413132000", // 광덕면
  "4413131000", // 풍세면
  "4413158000", // 청룡동
  "4413157000", // 신방동
] as const;

export type PilotRegionCode = (typeof PILOT_REGION_CODES)[number];

export function isPilotRegionCode(code: string): code is PilotRegionCode {
  return (PILOT_REGION_CODES as readonly string[]).includes(code);
}

export function regionName(code: string) {
  return REGION_BY_CODE[code]?.name ?? code;
}
