import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const [sourcePath, outputPath] = process.argv.slice(2);

if (!sourcePath || !outputPath) {
  throw new Error(
    "Usage: node scripts/extract-cheonan-subdivisions.mjs <source.geojson> <output.json>",
  );
}

const source = JSON.parse(await readFile(resolve(sourcePath), "utf8"));

function roundCoordinates(value) {
  if (typeof value === "number") return Number(value.toFixed(6));
  return value.map(roundCoordinates);
}

const features = source.features
  .filter((feature) => feature.properties.adm_nm.includes("천안시"))
  .map((feature) => {
    const { adm_cd2: code, adm_nm: fullName, sggnm } = feature.properties;
    const name = fullName.split(" ").at(-1);
    const district = sggnm.includes("동남구") ? "동남구" : "서북구";
    const kind = name.endsWith("읍") ? "읍" : name.endsWith("면") ? "면" : "동";

    return {
      type: "Feature",
      properties: { code, name, district, kind },
      geometry: {
        ...feature.geometry,
        coordinates: roundCoordinates(feature.geometry.coordinates),
      },
    };
  })
  .sort((a, b) => a.properties.code.localeCompare(b.properties.code));

if (features.length !== 31) {
  throw new Error(`Expected 31 Cheonan subdivisions, found ${features.length}`);
}

const output = {
  type: "FeatureCollection",
  features,
};

await writeFile(resolve(outputPath), `${JSON.stringify(output)}\n`, "utf8");
