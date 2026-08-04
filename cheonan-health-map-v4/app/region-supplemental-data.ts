export type RegionSupplementalRecord = {
  /** 2019년 독거노인 중 국민기초생활보장 수급권자 수 */
  basicLiving: number | null;
  /** 2019년 독거노인 중 저소득노인 수 */
  lowIncome: number | null;
  /** 2024년 의약품 등 판매업소 중 약국 수 */
  pharmacies: number | null;
};

/**
 * 기존 대시보드 원자료에 없던 항목만 별도 관리한다.
 * 불당1·2동의 2019년 독거노인 하위 구성은 당시 자료가 불당동으로만 집계되어
 * 2026년 65세 이상 인구 비율(불당1동 59.4%, 불당2동 40.6%)로 추정 배분한다.
 * 원자료의 불당동 합계와 맞도록 합계·하위항목을 각각 반올림 보정했다.
 */
export const REGION_SUPPLEMENTAL_BY_NAME: Record<string, RegionSupplementalRecord> = {
  목천읍: { basicLiving: 201, lowIncome: 68, pharmacies: 6 },
  풍세면: { basicLiving: 35, lowIncome: 37, pharmacies: 4 },
  광덕면: { basicLiving: 51, lowIncome: 22, pharmacies: 0 },
  북면: { basicLiving: 52, lowIncome: 54, pharmacies: 2 },
  성남면: { basicLiving: 28, lowIncome: 12, pharmacies: 0 },
  수신면: { basicLiving: 39, lowIncome: 15, pharmacies: 0 },
  병천면: { basicLiving: 83, lowIncome: 35, pharmacies: 5 },
  동면: { basicLiving: 27, lowIncome: 9, pharmacies: 0 },
  중앙동: { basicLiving: 129, lowIncome: 52, pharmacies: 12 },
  문성동: { basicLiving: 63, lowIncome: 13, pharmacies: 5 },
  원성1동: { basicLiving: 83, lowIncome: 10, pharmacies: 3 },
  원성2동: { basicLiving: 129, lowIncome: 11, pharmacies: 1 },
  봉명동: { basicLiving: 157, lowIncome: 55, pharmacies: 12 },
  일봉동: { basicLiving: 117, lowIncome: 84, pharmacies: 0 },
  신방동: { basicLiving: 99, lowIncome: 99, pharmacies: 20 },
  청룡동: { basicLiving: 292, lowIncome: 67, pharmacies: 22 },
  신안동: { basicLiving: 143, lowIncome: 32, pharmacies: 31 },
  성환읍: { basicLiving: 163, lowIncome: 26, pharmacies: 13 },
  성거읍: { basicLiving: 104, lowIncome: 58, pharmacies: 6 },
  직산읍: { basicLiving: 99, lowIncome: 84, pharmacies: 12 },
  입장면: { basicLiving: 84, lowIncome: 37, pharmacies: 6 },
  성정1동: { basicLiving: 239, lowIncome: 50, pharmacies: 11 },
  성정2동: { basicLiving: 214, lowIncome: 10, pharmacies: 13 },
  쌍용1동: { basicLiving: 43, lowIncome: 17, pharmacies: 26 },
  쌍용2동: { basicLiving: 46, lowIncome: 19, pharmacies: 8 },
  쌍용3동: { basicLiving: 298, lowIncome: 15, pharmacies: 5 },
  백석동: { basicLiving: 53, lowIncome: 22, pharmacies: 13 },
  불당1동: { basicLiving: 39, lowIncome: 4, pharmacies: 3 },
  불당2동: { basicLiving: 27, lowIncome: 2, pharmacies: 22 },
  부성1동: { basicLiving: 58, lowIncome: 9, pharmacies: 16 },
  부성2동: { basicLiving: 58, lowIncome: 18, pharmacies: 21 },
};
