const { readFileSync } = require("node:fs");
const { JSDOM } = require("jsdom");
const vm = require("node:vm");

const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>", { runScripts: "outside-only" });
const ctx = dom.getInternalVMContext();
ctx.structuredClone = global.structuredClone;
const mermaidCode = readFileSync("d:/RAG_DB_slim/mermaid.min.js", "utf8");
vm.runInContext(mermaidCode, ctx);
const mermaid = ctx.mermaid;
mermaid.initialize({ startOnLoad: false });

const html = readFileSync("d:/RAG_DB_slim/flowcharts.html", "utf8");
const blocks = [...html.matchAll(/<pre class="mermaid">([\s\S]*?)<\/pre>/g)];

(async () => {
  // Try the FULL block first
  try {
    await mermaid.parse(blocks[2][1]);
    console.log("Full block 2 PASS");
  } catch (e) {
    console.log("Full block 2 FAIL");
    console.log("hash:", JSON.stringify(e.hash));
    console.log("Trying subset starting from each line to isolate culprit");
    // Bisect by removing last N lines
    const lines = blocks[2][1].split("\n");
    let lo = 1, hi = lines.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      const sub = lines.slice(0, mid).join("\n");
      try {
        await mermaid.parse(sub);
        lo = mid + 1;
      } catch (e2) {
        hi = mid;
      }
    }
    console.log("First failing line number:", hi);
    console.log("Line content:", lines[hi - 1]);
    console.log("Previous line:", lines[hi - 2]);
    console.log("Next line:", lines[hi]);
  }
})();