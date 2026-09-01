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

(async () => {
  const tests = [
    { name: "comma separated class", code: "flowchart TD\nclassDef foo fill:#fff\nclass A,B,C foo" },
    { name: "comma Space class A,B,C foo", code: "flowchart TD\nclassDef foo fill:#fff\nclass A, B, C foo" },
    { name: "indented class a,b,c foo", code: "flowchart TD\nclassDef foo fill:#fff\nclass a,b,c foo\n  A[test]" },
    { name: "subgraph then class a,b,c foo", code: "flowchart TD\nsubgraph sg1\n  A --> B\nend\nclassDef foo fill:#fff\nclass A,B foo" }
  ];
  for (const t of tests) {
    try {
      await mermaid.parse(t.code);
      console.log(t.name + ": OK");
    } catch (e) {
      console.log(t.name + ": FAIL " + (e.hash ? "line " + e.hash.line + " col " + (e.hash.loc && e.hash.loc.first_column) : e.message));
    }
  }
})();