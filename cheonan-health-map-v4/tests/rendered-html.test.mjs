import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Cheonan map shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>천안시 행정구역별 고령·의료취약 지도<\/title>/i);
  assert.match(html, /천안시 행정구역별 고령·의료취약 지도/);
  assert.match(html, /전체 지도 보기/);
  assert.match(html, /읍·면·동 구분/);
  assert.match(html, /고령인구 밀도/);
  assert.match(html, /의료자원·사회취약 현황/);
  assert.match(html, /확인된 시설 세부정보 보기/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("includes all 31 Cheonan administrative subdivisions", async () => {
  const data = JSON.parse(
    await readFile(
      new URL("../app/cheonan-subdivisions.json", import.meta.url),
      "utf8",
    ),
  );
  const counts = Object.groupBy(
    data.features,
    (feature) => feature.properties.kind,
  );

  assert.equal(data.features.length, 31);
  assert.equal(counts.읍.length, 4);
  assert.equal(counts.면.length, 8);
  assert.equal(counts.동.length, 19);
});
