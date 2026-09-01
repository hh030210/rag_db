const { readFileSync } = require("node:fs");
const { JSDOM } = require("jsdom");
const vm = require("node:vm");

const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>", { runScripts: "outside-only" });
const ctx = dom.getInternalVMContext();
ctx.structuredClone = global.structuredClone;
const mermaidCode = readFileSync("d:/RAG_DB_slim/mermaid.min.js", "utf8");
vm.runInContext(mermaidCode, ctx);

const html = readFileSync("d:/RAG_DB_slim/flowcharts.html", "utf8");
const blocks = [...html.matchAll(/<pre class="mermaid">([\s\S]*?)<\/pre>/g)].map((m, i) => ({ i, code: m[1] }));
const mermaid = ctx.mermaid;
mermaid.initialize({ startOnLoad: false, theme: "default" });

(async () => {
  let pass = 0, fail = 0;
  for (const { i, code } of blocks) {
    try {
      await mermaid.parse(code);
      console.log(`Block ${i}: OK`);
      pass++;
    } catch (e) {
      console.log(`Block ${i}: FAIL`);
      console.log("str:", e.str);
      console.log("hash:", JSON.stringify(e.hash));
      fail++;
    }
  }
  console.log(`\nResult: ${pass} pass, ${fail} fail`);
  process.exit(fail ? 1 : 0);
})();