const fs = require("fs");
const b = fs.readFileSync("d:/RAG_DB_slim/flowcharts.html");
const s = b.toString("utf8");
// Look at line 35 (which corresponds to the broken line in block 2)
const lines = s.split("\n");
const outIdx = lines.findIndex(l => l.includes("Out --> ToSearch"));
console.log("Out line in file at index:", outIdx);
const line = lines[outIdx];
console.log("Full line (utf8):");
console.log(line);
console.log("\nByte-by-byte:");
for (let i = 0; i < line.length; i++) {
  const c = line.charCodeAt(i);
  if (c > 0x7f) {
    console.log("idx=" + i + " code=U+" + c.toString(16) + " [" + line[i] + "]");
  }
}