const { readFileSync } = require("node:fs");
const { JSDOM } = require("jsdom");
const vm = require("node:vm");

const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>", { runScripts: "outside-only", pretendToBeVisual: true });
const ctx = dom.getInternalVMContext();
ctx.structuredClone = global.structuredClone;
ctx.DOMPurify = require("dompurify")({ window: dom.window });
const mermaidCode = readFileSync("d:/RAG_DB_slim/mermaid.min.js", "utf8");
vm.runInContext(mermaidCode, ctx);

const html = readFileSync("d:/RAG_DB_slim/flowcharts.html", "utf8");
const blocks = [...html.matchAll(/<pre class="mermaid">([\s\S]*?)<\/pre>/g)].map((m, i) => ({ i, code: m[1] }));
const target = parseInt(process.argv[2] || "0", 10);
const mermaid = ctx.mermaid;
mermaid.initialize({ startOnLoad: false, theme: "default", flowchart: { useMaxWidth: true, htmlLabels: true } });

(async () => {
  try {
    const result = await mermaid.render(`m${target}`, blocks[target].code);
    console.log(`Block ${target}: OK`);
    console.log("svg length:", result.svg.length);
  } catch (e) {
    console.log(`Block ${target}: FAIL`);
    console.log("msg:", e.message);
    console.log("str:", e.str);
    console.log("hash:", JSON.stringify(e.hash, null, 2));
  }
})();