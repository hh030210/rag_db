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

// Inject a hook to capture preprocessed text
ctx.__captured = null;
mermaid = ctx.mermaid;
mermaid.initialize({ startOnLoad: false });

(async () => {
  try {
    await mermaid.parse(code);
    console.log("OK");
  } catch (e) {
    console.log("hash:", JSON.stringify(e.hash));
    // The mermaid internal 'sanitize' function or transformer may strip <br/>
    // Let's see what they pass to lexer by hooking run()
    const origLexer = ctx.mermaid.parse;
  }
  // Try to dump what mermaid actually lexes: monkey-patch the lexer
})();

console.log("Code length:", code.length);
console.log("Code has 'ToSearch[':", code.includes("ToSearch["));