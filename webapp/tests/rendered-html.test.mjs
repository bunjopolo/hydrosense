import assert from "node:assert/strict";
import test from "node:test";

test("server renders HydroSense without legacy hospital content", async () => {
  const { default: worker } = await import("../dist/server/index.js");
  const response = await worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /HydroSense/);
  assert.match(html, /Know what normal looks like/);
  assert.match(html, /Load synthetic demo/);
  assert.doesNotMatch(html, /hospital|SmartAirSense|TFT median forecast/i);
});
