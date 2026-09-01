const { readFileSync } = require("node:fs");
const { JSDOM } = require("jsdom");
const vm = require("node:vm");

const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>", { runScripts: "outside-only" });
const ctx = dom.getInternalVMContext();
ctx.structuredClone = global.structuredClone;
const mermaidCode = readFileSync("d:/RAG_DB_slim/mermaid.min.js", "utf8");
vm.runInContext(mermaidCode, ctx);

const html = readFileSync("d:/RAG_DB_slim/flowcharts.html", "utf8");
const blocks = [...html.matchAll(/<pre class="mermaid">([\s\S]*?)<\/pre>/g)];
const idx = parseInt(process.argv[2], 10);
const code = blocks[idx][1];
const lines = code.split("\n");
const start = parseInt(process.argv[3] || "1", 10);
const end = parseInt(process.argv[4] || lines.length.toString(), 10);
for (let i = start - 1; i < Math.min(lines.length, end); i++) {
  console.log((i + 1).toString().padStart(2, " ") + ": " + lines[i]);
}
console.log("\n(Total lines: " + lines.length + ")");
console.log("(Has " + (code.match(/<br\/>/g) || []).length + " <br/> tags)");