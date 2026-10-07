/** Read frozen numeric samples; create a two-sheet scientific Excel table.
 * Uses only the Codex bundled Node/artifact-tool runtime. No filter rerun,
 * ground-truth association, parameter selection or rounding of source values.
 */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaultTemp = path.join(root, "data", "tmp", "v16_second_table");
const threadId = "01a11067-689a-71a3-a0e4-e3f048b42212";
const argumentsMap = new Map();
for (let i=2; i<process.argv.length; i+=2) {
  assert(process.argv[i]?.startsWith("--") && process.argv[i+1], "Arguments require --key value pairs");
  argumentsMap.set(process.argv[i], process.argv[i+1]);
}
const inputPath = path.resolve(argumentsMap.get("--input") ?? path.join(root, "output", "v16_second_table", "second_samples.json"));
const outputPath = path.resolve(argumentsMap.get("--output") ?? path.join(root, "output", "v16_second_table", "outputs", threadId, "mmaud_1s_table.xlsx"));
const tempDir = path.resolve(argumentsMap.get("--temp") ?? defaultTemp);
const require = createRequire(path.join(tempDir, "package.json"));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve("@oai/artifact-tool")).href);
const inputBytes = await fs.readFile(inputPath);
const input = JSON.parse(inputBytes.toString("utf8"));
const methodOrder = ["radar_only", "radar_camera"];
const methodNames = { radar_only: "仅雷达", radar_camera: "雷达＋相机" };
const numericFields = [
  "time_since_initialization_s", "timestamp_s", "filtered_x_m", "filtered_y_m", "filtered_z_m",
  "filtered_Vx_mps", "filtered_Vy_mps", "filtered_Vz_mps", "filtered_V_mps",
  "P_xx_m2", "P_yy_m2", "P_zz_m2", "P_VxVx_m2ps2", "P_VyVy_m2ps2", "P_VzVz_m2ps2",
  "gt_x_m", "gt_y_m", "gt_z_m", "V_ref_x_mps", "V_ref_y_mps", "V_ref_z_mps", "V_ref_mps",
];
assert.equal(input.records.length, 20, "Expected exactly 20 preserved sample records");
const records = [...input.records].sort((a,b) => a.sample_index-b.sample_index || methodOrder.indexOf(a.method)-methodOrder.indexOf(b.method));
for (let i=0; i<20; i++) {
  const record=records[i];
  assert.equal(record.method, methodOrder[i%2]);
  assert.equal(record.sample_index, Math.floor(i/2));
  assert.equal(record.saved_tick_index, record.sample_index*100);
  assert(Math.abs(record.time_since_initialization_s-record.sample_index)<1e-9);
  for (const key of numericFields) assert(Number.isFinite(record[key]), `Missing finite ${key} at record ${i}`);
  for (const key of numericFields.filter(key=>key.startsWith("P_"))) assert(record[key]>=0, "Posterior variance must be nonnegative");
  assert(Math.abs(Math.hypot(record.filtered_Vx_mps,record.filtered_Vy_mps,record.filtered_Vz_mps)-record.filtered_V_mps)<1e-12);
  assert(Math.abs(Math.hypot(record.V_ref_x_mps,record.V_ref_y_mps,record.V_ref_z_mps)-record.V_ref_mps)<1e-12);
}

const wb=Workbook.create();
const position=wb.worksheets.add("Position");
const velocity=wb.worksheets.add("Velocity");
const headerRow=11, firstDataRow=12, lastDataRow=31;
const positionHeaders=["t (s)","方法","x (m)","y (m)","z (m)","x_ref (m)","y_ref (m)","z_ref (m)","Var(x) (m²)","Var(y) (m²)","Var(z) (m²)","t_log (s)"];
const velocityHeaders=["t (s)","方法","Vx (m/s)","Vy (m/s)","Vz (m/s)","V (m/s)","Vx_ref (m/s)","Vy_ref (m/s)","Vz_ref (m/s)","V_ref (m/s)","Var(Vx)\n(m²/s²)","Var(Vy)\n(m²/s²)","Var(Vz)\n(m²/s²)","t_log (s)"];
const positionRows=records.map(r=>[r.time_since_initialization_s,methodNames[r.method],r.filtered_x_m,r.filtered_y_m,r.filtered_z_m,r.gt_x_m,r.gt_y_m,r.gt_z_m,r.P_xx_m2,r.P_yy_m2,r.P_zz_m2,r.timestamp_s]);
const velocityRows=records.map(r=>[r.time_since_initialization_s,methodNames[r.method],r.filtered_Vx_mps,r.filtered_Vy_mps,r.filtered_Vz_mps,null,r.V_ref_x_mps,r.V_ref_y_mps,r.V_ref_z_mps,null,r.P_VxVx_m2ps2,r.P_VyVy_m2ps2,r.P_VzVz_m2ps2,r.timestamp_s]);

function formatSheet(sheet, lastCol, title, headers, rows, isVelocity) {
  const all=sheet.getRange(`A1:${lastCol}${lastDataRow}`);
  all.format.font={name:"Microsoft YaHei",size:10,color:"#17212D"};
  all.format.verticalAlignment="center";
  all.format.rowHeightPx=25;
  sheet.showGridLines=false;
  sheet.tabColor=isVelocity ? "#356383" : "#234A65";
  sheet.getRange("A2").values=[[title]];
  sheet.getRange("A2").format.font={name:"Microsoft YaHei",size:14,bold:true,color:"#17212D"};
  sheet.getRange(`A3:${lastCol}3`).format.borders={bottom:{style:"thin",color:"#9FB1BF"}};
  sheet.getRange("A4").values=[["来源：MMAUD V1 Mavic3。冻结1.6日志每秒样本，两方法各10条。"]];
  sheet.getRange("A5").values=[["坐标：雷达系。t为初始化后时间，t_log为原日志时间。数字保留源精度。"]];
  sheet.getRange("A6").values=[["Var为模型P后验对角，不等于实测误差方差；标定参数不确定度未完整传播。GT速度方差未知。该片段仍为探索性。"]];
  sheet.getRange("A7").values=[[isVelocity ? "V_ref由真实位置前后各0.5 s线性插值后中心差分得到，是1 s平均参考速度，非直接测速。" : "x_ref、y_ref、z_ref来自真实Leica位置插值及冻结坐标变换。位置单位m，方差单位m²。"]];
  sheet.getRange("A8").values=[[isVelocity ? "V_ref是1 s平均速度向量的模（净位移/1 s），不是路程平均速率。" : "状态和P来自冻结滤波后验，每秒采样，无重新运行或状态插值。"]];
  sheet.getRange("A9").values=[[isVelocity ? "仅雷达 t=0–4 s 的零速度为原保存输出与旧回波选点所致，不表示真实目标静止。" : "仅雷达 t=0–4 s 的重复位置来自原日志，不是补值；旧回波最近点策略导致停滞。"]];
  sheet.getRange("A10").values=[["MMAUD：https://ntu-aris.github.io/MMAUD/。ICRA2024，DOI 10.1109/ICRA57147.2024.10610957。"]];
  sheet.getRange(`A4:${lastCol}10`).format.font={name:"Microsoft YaHei",size:9,color:"#435366"};
  sheet.getRange(`A4:${lastCol}10`).format.wrapText=false;
  sheet.getRange(`A${headerRow}:${lastCol}${headerRow}`).values=[headers];
  sheet.getRange(`A${firstDataRow}:${lastCol}${lastDataRow}`).values=rows;
  sheet.getRange(`A${headerRow}:${lastCol}${headerRow}`).format={
    fill:"#234A65",font:{name:"Microsoft YaHei",size:10,bold:true,color:"#FFFFFF"},
    horizontalAlignment:"center",verticalAlignment:"center",wrapText:true,
    rowHeightPx:isVelocity ? 40 : 34,
    borders:{insideVertical:{style:"thin",color:"#FFFFFF"}},
  };
  sheet.getRange(`A${firstDataRow}:${lastCol}${lastDataRow}`).format.horizontalAlignment="right";
  sheet.getRange(`B${firstDataRow}:B${lastDataRow}`).format.horizontalAlignment="left";
  sheet.getRange(`A${firstDataRow}:A${lastDataRow}`).setNumberFormat("0");
  sheet.getRange(`${lastCol}${firstDataRow}:${lastCol}${lastDataRow}`).setNumberFormat("0.000000");
  if (isVelocity) {
    sheet.getRange(`C${firstDataRow}:J${lastDataRow}`).setNumberFormat("0.000");
    sheet.getRange(`K${firstDataRow}:M${lastDataRow}`).setNumberFormat("0.000000");
  } else {
    sheet.getRange(`C${firstDataRow}:H${lastDataRow}`).setNumberFormat("0.000");
    sheet.getRange(`I${firstDataRow}:K${lastDataRow}`).setNumberFormat("0.000000");
  }
  for (let second=0; second<10; second++) {
    const r=firstDataRow+second*2;
    sheet.getRange(`A${r}:${lastCol}${r+1}`).format.fill=second%2===0 ? "#EDF3F7" : "#FFFFFF";
    sheet.getRange(`A${r+1}:${lastCol}${r+1}`).format.borders={bottom:{style:"thin",color:"#D7E1E8"}};
  }
  sheet.getRange(`A1:A${lastDataRow}`).format.columnWidthPx=65;
  sheet.getRange(`B1:B${lastDataRow}`).format.columnWidthPx=112;
  sheet.getRange(`C1:${isVelocity?"J":"H"}${lastDataRow}`).format.columnWidthPx=isVelocity?102:92;
  sheet.getRange(`${isVelocity?"K":"I"}1:${isVelocity?"M":"K"}${lastDataRow}`).format.columnWidthPx=112;
  sheet.getRange(`${lastCol}1:${lastCol}${lastDataRow}`).format.columnWidthPx=96;
  sheet.getRange("A1").format.rowHeightPx=14;
  sheet.getRange("A2").format.rowHeightPx=30;
  sheet.getRange("A3").format.rowHeightPx=10;
  sheet.freezePanes.freezeRows(headerRow);
  sheet.freezePanes.freezeColumns(2);
}

formatSheet(position,"L","MMAUD 每秒位置与模型后验方差",positionHeaders,positionRows,false);
formatSheet(velocity,"N","MMAUD 每秒速度与模型后验方差",velocityHeaders,velocityRows,true);
velocity.getRange(`F${firstDataRow}`).formulas=[[`=SQRT(C${firstDataRow}^2+D${firstDataRow}^2+E${firstDataRow}^2)`]];
velocity.getRange(`F${firstDataRow}:F${lastDataRow}`).fillDown();
velocity.getRange(`J${firstDataRow}`).formulas=[[`=SQRT(G${firstDataRow}^2+H${firstDataRow}^2+I${firstDataRow}^2)`]];
velocity.getRange(`J${firstDataRow}:J${lastDataRow}`).fillDown();

// Required final recalc, exact source comparisons and formula inspection.
wb.recalculate();
const positionValues=position.getRange(`A${firstDataRow}:L${lastDataRow}`).values;
const velocityValues=velocity.getRange(`A${firstDataRow}:N${lastDataRow}`).values;
for (let r=0;r<20;r++) {
  assert.deepEqual(positionValues[r],positionRows[r]);
  for (let c=0;c<14;c++) {
    const expected=c===5 ? records[r].filtered_V_mps : c===9 ? records[r].V_ref_mps : velocityRows[r][c];
    if (c===5 || c===9) assert(Math.abs(velocityValues[r][c]-expected)<1e-12);
    else assert.equal(velocityValues[r][c],expected);
  }
}
assert.equal(velocity.getRange(`F${lastDataRow}`).formulas[0][0],`=SQRT(C${lastDataRow}^2+D${lastDataRow}^2+E${lastDataRow}^2)`);
assert.equal(velocity.getRange(`J${lastDataRow}`).formulas[0][0],`=SQRT(G${lastDataRow}^2+H${lastDataRow}^2+I${lastDataRow}^2)`);
const inspections=[];
for (const name of ["Position","Velocity"]) {
  inspections.push((await wb.inspect({kind:"table",range:`${name}!A${headerRow}:${name==="Position"?"L":"N"}${headerRow+4}`,include:"values,formulas",tableMaxRows:5,tableMaxCols:14,maxChars:7000})).ndjson);
}
const errors=await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:100},summary:"Final formula error scan"});
inspections.push(errors.ndjson);
await fs.mkdir(tempDir,{recursive:true});
await fs.writeFile(path.join(tempDir,"inspection.ndjson"),inspections.join("\n"));
assert(!/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson),"Unexpected Excel formula error");
for (const name of ["Position","Velocity"]) {
  const preview=await wb.render({sheetName:name,range:`A1:${name==="Position"?"L":"N"}${lastDataRow}`,scale:1,format:"png"});
  await fs.writeFile(path.join(tempDir,`${name.toLowerCase()}_preview.png`),new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(path.dirname(outputPath),{recursive:true});
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(outputPath);
// The runtime writes an export inspection sidecar. Keep support files in tmp.
try {
  await fs.copyFile(`${outputPath}.inspect.ndjson`,path.join(tempDir,"export_inspection.ndjson"));
  await fs.unlink(`${outputPath}.inspect.ndjson`);
} catch(error) { if(error.code!=="ENOENT") throw error; }
await fs.writeFile(path.join(tempDir,"build_check.json"),JSON.stringify({
  input_sha256:crypto.createHash("sha256").update(inputBytes).digest("hex"),
  output:outputPath,worksheets:["Position","Velocity"],records_per_sheet:20,
  preserved_source_precision:true,formula_magnitudes_checked_against_json:true,
  formula_fill_last_row_checked:true,recalculated:true,formula_error_inspect:errors.ndjson,
  native_excel_recalculation_tested:false,
},null,2));
console.log(JSON.stringify({output:outputPath,recordsPerSheet:20,checks:"full-precision source values, V/V_ref formulas, last-row references, recalc/error scan, two previews",previews:tempDir}));
