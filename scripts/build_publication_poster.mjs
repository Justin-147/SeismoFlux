import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const PACKAGE_ROOT = path.join(
  REPO_ROOT,
  "outputs",
  "publication",
  "seismoflux_b0_r30_v1",
);
const FIGURE_DIR = path.join(PACKAGE_ROOT, "figures");

const DEFAULT_OUTPUT = path.join(
  PACKAGE_ROOT,
  "poster",
  "seismoflux_scientific_poster_a0_landscape.pptx",
);

const W = 4493;
const H = 3179;
const FONT_CN = "Microsoft YaHei";
const FONT_LATIN = "Arial";

const C = {
  navy: "#0B2638",
  navy2: "#163E54",
  blue: "#1769AA",
  bluePale: "#EAF3F8",
  teal: "#23847E",
  coral: "#E46F51",
  coralPale: "#FCEEE9",
  ink: "#142B3A",
  muted: "#60717C",
  light: "#F5F8F8",
  rule: "#C9D7DE",
  white: "#FFFFFF",
  gray: "#8A979F",
};

function parseOutputPath() {
  const args = process.argv.slice(2);
  const outputIndex = args.indexOf("--output");
  if (outputIndex >= 0) {
    if (!args[outputIndex + 1]) {
      throw new Error("--output requires a .pptx path");
    }
    return path.resolve(args[outputIndex + 1]);
  }
  const positional = args.find((value) => !value.startsWith("-"));
  return positional ? path.resolve(positional) : DEFAULT_OUTPUT;
}

async function writeBlob(targetPath, blob) {
  await fs.writeFile(targetPath, new Uint8Array(await blob.arrayBuffer()));
}

function addRect(slide, name, left, top, width, height, fill, options = {}) {
  return slide.shapes.add({
    geometry: options.rounded ? "roundRect" : "rect",
    name,
    position: { left, top, width, height },
    fill,
    line: {
      style: "solid",
      fill: options.lineFill ?? "none",
      width: options.lineWidth ?? 0,
    },
    ...(options.rounded ? { borderRadius: options.radius ?? 18 } : {}),
  });
}

function addText(slide, {
  name,
  text,
  left,
  top,
  width,
  height,
  fontSize,
  color = C.ink,
  bold = false,
  typeface = FONT_CN,
  alignment = "left",
  verticalAlignment = "top",
  lineSpacing = 1.08,
  insets = { top: 0, right: 0, bottom: 0, left: 0 },
}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize,
    color,
    bold,
    typeface,
    alignment,
    verticalAlignment,
    lineSpacing,
    autoFit: "shrinkText",
    wrap: "square",
    insets,
  };
  return box;
}

function addSectionTitle(slide, name, title, left, top, width) {
  addText(slide, {
    name: `${name}-title`,
    text: title,
    left,
    top,
    width,
    height: 74,
    fontSize: 53,
    color: C.navy,
    bold: true,
  });
  addRect(slide, `${name}-accent`, left, top + 76, 132, 9, C.coral);
  addRect(slide, `${name}-rule`, left + 148, top + 80, width - 148, 2, C.rule);
}

async function addSvgFigure(slide, name, filename, alt, position) {
  const figurePath = path.join(FIGURE_DIR, filename);
  const svg = await fs.readFile(figurePath, "utf8");
  addRect(
    slide,
    `${name}-backdrop`,
    position.left,
    position.top,
    position.width,
    position.height,
    C.white,
    { lineFill: C.rule, lineWidth: 2, rounded: true, radius: 12 },
  );
  slide.images.add({
    svg,
    alt,
    fit: "contain",
    position: {
      left: position.left + 10,
      top: position.top + 10,
      width: position.width - 20,
      height: position.height - 20,
    },
  });
}

function addBulletBlock(slide, name, items, position, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text.set(
    items.map((item) => ({
      bulletCharacter: "•",
      marginLeft: 34,
      indent: -22,
      spaceAfter: options.spaceAfter ?? 12,
      runs: item,
    })),
  );
  box.text.style = {
    fontSize: options.fontSize ?? 33,
    color: options.color ?? C.ink,
    typeface: FONT_CN,
    lineSpacing: options.lineSpacing ?? 1.13,
    autoFit: "shrinkText",
    wrap: "square",
    insets: { top: 0, right: 4, bottom: 0, left: 0 },
  };
  return box;
}

async function buildPoster(outputPath) {
  const presentation = Presentation.create({
    slideSize: { width: W, height: H },
  });
  const slide = presentation.slides.add();
  slide.background.fill = C.light;

  // Header: the scientific conclusion is visible before any methodological detail.
  addRect(slide, "header", 0, 0, W, 420, C.navy);
  addRect(slide, "header-accent", 0, 0, 26, 420, C.coral);
  addText(slide, {
    name: "header-kicker",
    text: "SEISMOFLUX  ·  HISTORICAL DEVELOPMENT EVIDENCE",
    left: 140,
    top: 54,
    width: 2800,
    height: 44,
    fontSize: 31,
    color: "#BFD3DE",
    bold: true,
    typeface: FONT_LATIN,
  });
  addText(slide, {
    name: "poster-title",
    text: "相同报警面积上限下，近期地震活动可提高\n规则化 M5–6 震群召回",
    left: 140,
    top: 104,
    width: 2910,
    height: 215,
    fontSize: 108,
    color: C.white,
    bold: true,
    lineSpacing: 0.94,
  });
  addText(slide, {
    name: "poster-subtitle",
    text: "75 km 地震活动背景（B0）与近期 30 天 M4+ 信息融合（B0_R30）的冻结历史评估",
    left: 145,
    top: 334,
    width: 2850,
    height: 50,
    fontSize: 31,
    color: "#D8E6EC",
  });

  addText(slide, {
    name: "header-metric",
    text: "5 / 21  →  9 / 21",
    left: 3130,
    top: 92,
    width: 1215,
    height: 150,
    fontSize: 123,
    color: C.white,
    bold: true,
    typeface: FONT_LATIN,
    alignment: "center",
    verticalAlignment: "middle",
  });
  addText(slide, {
    name: "header-gain",
    text: "+19.05 个百分点",
    left: 3190,
    top: 245,
    width: 1095,
    height: 78,
    fontSize: 54,
    color: "#FFB39F",
    bold: true,
    alignment: "center",
    verticalAlignment: "middle",
  });
  addText(slide, {
    name: "header-area",
    text: "相同 600,000 km² 面积上限",
    left: 3190,
    top: 326,
    width: 1095,
    height: 50,
    fontSize: 30,
    color: "#D8E6EC",
    alignment: "center",
  });

  const leftX = 140;
  const middleX = 1490;
  const rightX = 3080;
  const leftW = 1270;
  const middleW = 1510;
  const rightW = 1270;

  addRect(slide, "column-divider-left", 1448, 485, 3, 2470, C.rule);
  addRect(slide, "column-divider-right", 3038, 485, 3, 2470, C.rule);

  // Left column: question, data, and the frozen model.
  addSectionTitle(slide, "left-column", "问题、数据与冻结方法", leftX, 470, leftW);
  addRect(slide, "question-callout", leftX, 570, leftW, 214, C.bluePale, {
    rounded: true,
    radius: 18,
  });
  addText(slide, {
    name: "question-callout-title",
    text: "科学问题",
    left: leftX + 32,
    top: 594,
    width: 300,
    height: 46,
    fontSize: 35,
    color: C.blue,
    bold: true,
  });
  addText(slide, {
    name: "question-callout-body",
    text: "在报警面积上限不提高的前提下，近期地震活动能否帮助模型找到更多未来 30 天内的规则化 M5–6 震群？",
    left: leftX + 32,
    top: 646,
    width: leftW - 64,
    height: 112,
    fontSize: 34,
    color: C.ink,
    bold: true,
    lineSpacing: 1.08,
  });

  addText(slide, {
    name: "data-heading",
    text: "用于本次评估的数据",
    left: leftX,
    top: 818,
    width: leftW,
    height: 52,
    fontSize: 39,
    color: C.navy,
    bold: true,
  });
  addBulletBlock(
    slide,
    "data-bullets",
    [
      [
        { run: "40,898", textStyle: { bold: true, typeface: FONT_LATIN, color: C.blue } },
        " 个去重地震事件；所有特征严格限定在起报时刻之前",
      ],
      [
        { run: "15,697", textStyle: { bold: true, typeface: FONT_LATIN, color: C.blue } },
        " 个约 25 km 网格；候选位置不依赖真实震中",
      ],
      [
        { run: "21", textStyle: { bold: true, typeface: FONT_LATIN, color: C.blue } },
        " 个按 75 km / 30 天规则合并的 M5–6 震群；降低同一序列重复计数",
      ],
      ["动态异常资料曾单独检验，但未提高主终点，因此未进入最终融合分数"],
    ],
    { left: leftX, top: 880, width: leftW, height: 294 },
    { fontSize: 31, spaceAfter: 10 },
  );

  addText(slide, {
    name: "method-heading",
    text: "冻结模型：B0_R30",
    left: leftX,
    top: 1192,
    width: leftW,
    height: 54,
    fontSize: 39,
    color: C.navy,
    bold: true,
  });
  await addSvgFigure(
    slide,
    "method-figure",
    "figure_05_method_overview.svg",
    "SeismoFlux 冻结 B0_R30 方法流程：仅使用起报时刻之前的数据，在固定面积预算内排序候选网格。",
    { left: leftX, top: 1254, width: leftW, height: 558 },
  );
  addText(slide, {
    name: "method-caption",
    text: "75 km 平滑背景描述长期地震活动；近期 30 天 M4+ 活动用于重新排序。输出是相对强度和顺位，不是绝对发震概率。",
    left: leftX + 4,
    top: 1822,
    width: leftW - 8,
    height: 112,
    fontSize: 29,
    color: C.muted,
    lineSpacing: 1.08,
  });

  addRect(slide, "area-callout", leftX, 1960, leftW, 252, C.coralPale, {
    rounded: true,
    radius: 18,
  });
  addText(slide, {
    name: "area-callout-title",
    text: "公平比较：报警面积上限始终固定",
    left: leftX + 30,
    top: 1982,
    width: leftW - 60,
    height: 48,
    fontSize: 36,
    color: "#A8432D",
    bold: true,
  });
  addText(slide, {
    name: "area-callout-body",
    text: "两个模型都受 600,000 km² 上限约束；平均实际面积为 599,447 与 599,666 km²，差异小于一个网格。召回增加不是因为“画了更多区域”。",
    left: leftX + 30,
    top: 2042,
    width: leftW - 60,
    height: 138,
    fontSize: 32,
    color: C.ink,
    lineSpacing: 1.1,
  });

  addText(slide, {
    name: "design-heading",
    text: "评价设计",
    left: leftX,
    top: 2240,
    width: leftW,
    height: 52,
    fontSize: 39,
    color: C.navy,
    bold: true,
  });
  addBulletBlock(
    slide,
    "design-bullets",
    [
      ["按时间顺序训练与验证，禁止使用起报日之后的信息"],
      ["30 天为主评价窗；90 天结果仅作为补充"],
      ["三个严格时间外推折，并用 2,000 次配对震群自助法估计不确定性"],
      ["检验对象是相同报警面积上限下的规则化震群召回"],
    ],
    { left: leftX, top: 2300, width: leftW, height: 335 },
    { fontSize: 31, spaceAfter: 13 },
  );

  addText(slide, {
    name: "anomaly-heading",
    text: "异常资料的结论",
    left: leftX,
    top: 2660,
    width: leftW,
    height: 52,
    fontSize: 39,
    color: C.navy,
    bold: true,
  });
  addText(slide, {
    name: "anomaly-body",
    text: "205 期、59,904 条异常观测已重建为约 322 万条特征并完成检验，但尚无主终点增益证据。当前结论来自地震目录中的长期背景与近期活动。",
    left: leftX,
    top: 2720,
    width: leftW,
    height: 226,
    fontSize: 30,
    color: C.muted,
    lineSpacing: 1.12,
  });

  // Middle column: primary endpoint and all-cluster evidence.
  addSectionTitle(slide, "middle-column", "主要结果：相同面积，多召回 4 个震群", middleX, 470, middleW);
  addText(slide, {
    name: "metric-baseline",
    text: "5 / 21",
    left: middleX + 40,
    top: 588,
    width: 405,
    height: 132,
    fontSize: 111,
    color: C.gray,
    bold: true,
    typeface: FONT_LATIN,
    alignment: "center",
  });
  addText(slide, {
    name: "metric-arrow",
    text: "→",
    left: middleX + 452,
    top: 592,
    width: 165,
    height: 128,
    fontSize: 101,
    color: C.rule,
    bold: true,
    typeface: FONT_LATIN,
    alignment: "center",
  });
  addText(slide, {
    name: "metric-r30",
    text: "9 / 21",
    left: middleX + 620,
    top: 588,
    width: 420,
    height: 132,
    fontSize: 111,
    color: C.blue,
    bold: true,
    typeface: FONT_LATIN,
    alignment: "center",
  });
  addText(slide, {
    name: "metric-gain",
    text: "+4",
    left: middleX + 1085,
    top: 575,
    width: 305,
    height: 128,
    fontSize: 96,
    color: C.coral,
    bold: true,
    typeface: FONT_LATIN,
    alignment: "center",
  });
  addText(slide, {
    name: "metric-baseline-label",
    text: "B0 背景模型",
    left: middleX + 40,
    top: 720,
    width: 405,
    height: 42,
    fontSize: 28,
    color: C.muted,
    alignment: "center",
  });
  addText(slide, {
    name: "metric-r30-label",
    text: "B0_R30 冻结模型",
    left: middleX + 620,
    top: 720,
    width: 420,
    height: 42,
    fontSize: 28,
    color: C.blue,
    bold: true,
    alignment: "center",
  });
  addText(slide, {
    name: "metric-gain-label",
    text: "+19.05 pp",
    left: middleX + 1085,
    top: 710,
    width: 305,
    height: 50,
    fontSize: 32,
    color: C.coral,
    bold: true,
    typeface: FONT_LATIN,
    alignment: "center",
  });

  await addSvgFigure(
    slide,
    "primary-result-figure",
    "figure_01_primary_result.svg",
    "B0 与 B0_R30 在不同报警面积上限下的规则化 M5–6 震群召回，以及三折结果。",
    { left: middleX, top: 790, width: middleW, height: 828 },
  );
  addText(slide, {
    name: "primary-result-caption",
    text: "在 600,000 km² 主评价上限，配对震群自助法 95% 区间为 +4.76 至 +38.10 个百分点；正增益复本比例为 0.9905，不是模型正确概率。",
    left: middleX + 4,
    top: 1628,
    width: middleW - 8,
    height: 90,
    fontSize: 29,
    color: C.muted,
    lineSpacing: 1.08,
  });

  addText(slide, {
    name: "all-clusters-heading",
    text: "21 个规则化震群逐一核对",
    left: middleX,
    top: 1746,
    width: middleW,
    height: 58,
    fontSize: 41,
    color: C.navy,
    bold: true,
  });
  await addSvgFigure(
    slide,
    "all-cluster-figure",
    "figure_02_all_cluster_outcomes.svg",
    "21 个规则化 M5–6 震群在 B0 与 B0_R30 下的命中结果：5 个共同命中、4 个新增命中、0 个损失、12 个共同未命中。",
    { left: middleX + 90, top: 1814, width: middleW - 180, height: 988 },
  );
  addText(slide, {
    name: "all-cluster-caption",
    text: "逐一结果为：5 个共同命中、4 个新增命中、0 个损失、12 个共同未命中。改进集中在前两折；Fold 3 为 0/7→0/7。",
    left: middleX + 4,
    top: 2816,
    width: middleW - 8,
    height: 110,
    fontSize: 29,
    color: C.muted,
    lineSpacing: 1.08,
  });

  // Right column: interpretable event examples and limits.
  addSectionTitle(slide, "right-column", "代表震例与科学边界", rightX, 470, rightW);
  await addSvgFigure(
    slide,
    "case-study-figure",
    "figure_03_case_studies.svg",
    "两个代表性新增命中震例：2024 年 10 月库车 M5.5 与 2023 年 12 月肃北 M5.0。",
    { left: rightX, top: 570, width: rightW, height: 875 },
  );
  addText(slide, {
    name: "case-study-caption",
    text: "地图和时间线只展示起报时刻之前可见的地震活动，以及后来用于评价的目标震群；真实震中未参与候选位置生成。",
    left: rightX + 4,
    top: 1455,
    width: rightW - 8,
    height: 94,
    fontSize: 28,
    color: C.muted,
    lineSpacing: 1.08,
  });

  addText(slide, {
    name: "case-highlights-heading",
    text: "为什么这两个震例直观",
    left: rightX,
    top: 1576,
    width: rightW,
    height: 52,
    fontSize: 39,
    color: C.navy,
    bold: true,
  });
  addBulletBlock(
    slide,
    "case-highlights",
    [
      [
        { run: "库车 M5.5：", textStyle: { bold: true, color: C.blue } },
        "目标网格顺位由 1,725 升至第 3；起报前约 4 小时附近出现 M4.9。",
      ],
      [
        { run: "肃北 M5.0：", textStyle: { bold: true, color: C.blue } },
        "顺位由 3,912 升至第 194；起报前约 29 天、14.5 km 内有 M5.5。",
      ],
    ],
    { left: rightX, top: 1640, width: rightW, height: 244 },
    { fontSize: 31, spaceAfter: 15 },
  );

  await addSvgFigure(
    slide,
    "rank-shift-figure",
    "figure_04_case_rank_shifts.svg",
    "四个新增命中震群的目标网格顺位变化，包括肃北、杂多和两次库车震群。",
    { left: rightX, top: 1902, width: rightW, height: 645 },
  );
  addText(slide, {
    name: "rank-shift-caption",
    text: "四个新增命中均发生在西北区域（Fold 1–2）；两次库车目标位于同一个约 25 km 网格，因此不能把它们当作两个单独的新空间发现。",
    left: rightX + 4,
    top: 2557,
    width: rightW - 8,
    height: 118,
    fontSize: 28,
    color: C.muted,
    lineSpacing: 1.08,
  });

  addText(slide, {
    name: "limits-heading",
    text: "结论成立到哪里",
    left: rightX,
    top: 2694,
    width: rightW,
    height: 52,
    fontSize: 39,
    color: C.navy,
    bold: true,
  });
  addBulletBlock(
    slide,
    "limits-bullets",
    [
      ["这是冻结模型的历史开发证据，不是真实前瞻结论"],
      ["Fold 3 没有增益，说明效果存在明显地域差异"],
      ["地理分块不确定性尚未纳入现有震群自助区间"],
      ["真实前瞻已授权并冻结；截至2026年8月31日首期尚未起报，目前为0期"],
    ],
    { left: rightX, top: 2750, width: rightW, height: 218 },
    { fontSize: 28, spaceAfter: 8, lineSpacing: 1.07 },
  );

  // Footer states the interpretation boundary in audience-facing language.
  addRect(slide, "footer", 0, 3020, W, 159, "#E5EDF1");
  addText(slide, {
    name: "footer-note",
    text: "核心意义：近期地震活动在固定报警面积上限下提供了可解释的历史增益；能否迁移到未来，必须由不可回写的真实前瞻结果回答。",
    left: 140,
    top: 3052,
    width: 3500,
    height: 74,
    fontSize: 32,
    color: C.navy,
    bold: true,
    verticalAlignment: "middle",
  });
  addText(slide, {
    name: "footer-version",
    text: "冻结协议：P1 v0.2.7  ·  30 d 主评价",
    left: 3660,
    top: 3052,
    width: 690,
    height: 74,
    fontSize: 27,
    color: C.muted,
    typeface: FONT_CN,
    alignment: "right",
    verticalAlignment: "middle",
  });

  slide.speakerNotes.textFrame.setText([
    "[Sources]",
    "Frozen primary metrics: outputs/publication/seismoflux_b0_r30_v1/figure_data/primary_metrics.csv",
    "All-cluster outcomes: outputs/publication/seismoflux_b0_r30_v1/figure_data/cluster_outcomes_30d_600k.csv",
    "Illustrative cases: outputs/publication/seismoflux_b0_r30_v1/figure_data/illustrative_case_summary.csv",
    "Embedded scientific figures: outputs/publication/seismoflux_b0_r30_v1/figures/figure_01_primary_result.svg through figure_05_method_overview.svg",
    "No external visual assets are used.",
    "[/Sources]",
  ]);

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const qaDir = path.join(path.dirname(outputPath), "artifact_tool_qa");
  await fs.mkdir(qaDir, { recursive: true });
  const preview = await presentation.export({
    slide,
    format: "png",
    scale: 0.25,
  });
  await writeBlob(path.join(qaDir, "poster-preview.png"), preview);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(
    path.join(qaDir, "poster.layout.json"),
    await layout.text(),
    "utf8",
  );
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outputPath);
}

const outputPath = parseOutputPath();
buildPoster(outputPath).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
