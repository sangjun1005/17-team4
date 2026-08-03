import cheonanAgeCsv from "./data/cheonan-age-population-2026-06.csv?raw";

type CheonanAgeRecord = {
  regionCode: string;
  regionName: string;
  period: string;
  totalPopulation: number;
  age65To69: number;
  age70To74: number;
  age75To79: number;
  age80To84: number;
  age85Plus: number;
  age80Plus: number;
};

export type CheonanAgeProfile = {
  period: string;
  totalPopulation: number;
  ageBands: [number, number, number, number];
  ageComposition: [number, number, number];
};

function parseCsv(csv: string): CheonanAgeRecord[] {
  const [, ...lines] = csv.trim().split(/\r?\n/);
  return lines.filter(Boolean).map((line) => {
    const [regionCode, regionName, period, totalPopulation, age65To69, age70To74, age75To79, age80To84, age85Plus, age80Plus] = line.split(",");
    return {
      regionCode,
      regionName,
      period,
      totalPopulation: Number(totalPopulation),
      age65To69: Number(age65To69),
      age70To74: Number(age70To74),
      age75To79: Number(age75To79),
      age80To84: Number(age80To84),
      age85Plus: Number(age85Plus),
      age80Plus: Number(age80Plus),
    };
  });
}

export const CHEONAN_AGE_BY_NAME: Record<string, CheonanAgeProfile> = Object.fromEntries(
  parseCsv(cheonanAgeCsv).map((row) => [
    row.regionName,
    {
      period: row.period,
      totalPopulation: row.totalPopulation,
      ageBands: [row.age65To69, row.age70To74, row.age75To79, row.age80Plus],
      ageComposition: [
        row.age65To69 + row.age70To74,
        row.age75To79 + row.age80To84,
        row.age85Plus,
      ],
    },
  ]),
) as Record<string, CheonanAgeProfile>;
