/** Excel-export helper for litesizer_zeta_potential.py. Receives JSON on stdin.
 * Uses the installed spreadsheet runtime; creates only the requested workbook.
 * Set LITESIZER_EXCEL_QA_DIR to render temporary PNG previews for visual checks.
 */
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const input = JSON.parse(Buffer.concat(chunks).toString("utf8"));
const runtime = process.argv[2];
const scratch = await fs.mkdtemp(path.join(os.tmpdir(), "litesizer-excel-"));
try {
  await fs.symlink(path.join(runtime, "node_modules"), path.join(scratch, "node_modules"), "junction");
  const require = createRequire(path.join(scratch, "export.mjs"));
  const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve("@oai/artifact-tool")));
  const book = Workbook.create();

  function table(name, data) {
    const sheet = book.worksheets.add(name);
    const matrix = [data.columns, ...data.data];
    const range = sheet.getRangeByIndexes(0, 0, matrix.length, data.columns.length);
    range.values = matrix;
    range.format.font = { name: "Open Sans", size: 10 };
    range.format.rowHeight = 22;
    range.format.columnWidth = 24;
    range.format.verticalAlignment = "center";
    const header = sheet.getRangeByIndexes(0, 0, 1, data.columns.length);
    header.format = {
      fill: "#000099", font: { name: "Open Sans", size: 10, bold: true, color: "#FFFFFF" },
      wrapText: true, rowHeight: 46, horizontalAlignment: "center", verticalAlignment: "center",
    };
    for (let column = 0; column < data.columns.length; column++) {
      const key = data.columns[column];
      const body = sheet.getRangeByIndexes(1, column, data.data.length, 1);
      if (data.data.some(row => typeof row[column] === "number")) {
        body.setNumberFormat(key.startsWith("n_") || key === "nominal_size_nm" ? "0" : "0.000");
        body.format.horizontalAlignment = "right";
      }
      if (key.includes("source") || key.includes("measurement_names")) {
        body.format.columnWidth = 55;
        body.format.wrapText = true;
      }
    }
    range.format.autofitRows();
    header.format.rowHeight = 46;
    sheet.freezePanes.freezeRows(1);
    sheet.freezePanes.freezeColumns(1);
    sheet.showGridLines = false;
    return sheet;
  }

  table("Statistics", input.summary);
  table("Individual measurements", input.individual);
  const caption = book.worksheets.add("Caption");
  const paragraphs = input.caption.trim().split(/\n\n/);
  const captionRange = caption.getRangeByIndexes(0, 0, paragraphs.length + 2, 1);
  captionRange.values = [["Zeta potential: caption and interpretation"], [""], ...paragraphs.map(p => [p])];
  captionRange.format.font = { name: "Open Sans", size: 10 };
  captionRange.format.columnWidth = 110;
  captionRange.format.wrapText = true;
  captionRange.format.verticalAlignment = "top";
  captionRange.format.autofitRows();
  caption.getRange("A1").format.font = { name: "Open Sans", size: 14, bold: true };
  caption.getRange("A1").format.rowHeight = 28;
  caption.showGridLines = false;

  book.recalculate();
  const errors = await book.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#N/A",
    options: { useRegex: true, maxResults: 20 }, maxChars: 1000 });
  console.log(errors.ndjson);
  if (process.env.LITESIZER_EXCEL_QA_DIR) {
    const qa = process.env.LITESIZER_EXCEL_QA_DIR;
    await fs.mkdir(qa, { recursive: true });
    for (const [name, range] of [["Statistics", "A1:L7"], ["Individual measurements", "A1:L9"], ["Caption", "A1:A12"]]) {
      const preview = await book.render({ sheetName: name, range, scale: 1, format: "png" });
      await fs.writeFile(path.join(qa, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
    }
  }
  const result = await SpreadsheetFile.exportXlsx(book);
  const temporaryWorkbook = path.join(scratch, "results.xlsx");
  await result.save(temporaryWorkbook);
  await fs.copyFile(temporaryWorkbook, input.output);
} finally {
  // Remove only this helper's own temporary directory, never its dependency target.
  if (path.dirname(scratch) === os.tmpdir() && path.basename(scratch).startsWith("litesizer-excel-")) {
    await fs.unlink(path.join(scratch, "results.xlsx")).catch(() => {});
    await fs.unlink(path.join(scratch, "results.xlsx.inspect.ndjson")).catch(() => {});
    await fs.unlink(path.join(scratch, "node_modules")).catch(() => {});
    await fs.rmdir(scratch).catch(() => {});
  }
}
