import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const PUBLICATION_ROOT = path.join(
  REPO_ROOT,
  "outputs",
  "publication",
  "seismoflux_b0_r30_v1",
);
const FIGURE_DIR = path.join(PUBLICATION_ROOT, "figures");
const DEFAULT_OUTPUT_DIR = path.join(PUBLICATION_ROOT, "presentation");

const COLORS = {
  canvas: "#FFFFFF",
  ink: "#102A43",
  muted: "#526777",
  faint: "#E8EEF2",
  panel: "#F4F7F8",
  blue: "#1769AA",
  blueLight: "#DDEEF8",
  orange: "#E07A3F",
  orangeLight: "#FBE8DC",
  green: "#138A72",
  greenLight: "#DDF2EC",
  red: "#B5463A",
};

const FONTS = {
  cn: "Microsoft YaHei",
  latin: "Arial",
};

const SOURCE_PATHS = {
  figureCaptions:
    "outputs/publication/seismoflux_b0_r30_v1/FIGURE_CAPTIONS.md",
  metrics:
    "outputs/publication/seismoflux_b0_r30_v1/figure_data/primary_metrics.csv",
  clusters:
    "outputs/publication/seismoflux_b0_r30_v1/figure_data/cluster_outcomes_30d_600k.csv",
  cases:
    "outputs/publication/seismoflux_b0_r30_v1/figure_data/illustrative_case_summary.csv",
  payload:
    "outputs/publication/seismoflux_b0_r30_v1/build/science_payload.json",
  d1Acceptance: "docs/d1_final_acceptance_2026-08-28.md",
  p1Protocol: "docs/p1_b0_r30_prospective_preregistration.md",
  p1Authorization: "docs/p1_real_issue_authorization_acceptance_2026-08-31.md",
  etasQualification: "docs/phase2_etas_numerical_qualification_acceptance.md",
  sourceBoundary: "data/manifests/p1_source_boundary_manifest.json",
};

function notesBlock(lines) {
  return [
    "[Sources]",
    ...lines.map((line) => `- ${line}`),
    "[/Sources]",
  ].join("\n");
}

function addNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(notesBlock(lines));
  slide.speakerNotes.setVisible(true);
}

function addText(
  slide,
  {
    name,
    text,
    left,
    top,
    width,
    height,
    fontSize = 22,
    color = COLORS.ink,
    bold = false,
    typeface = FONTS.cn,
    alignment = "left",
    verticalAlignment = "top",
    fill = "none",
    lineFill = "none",
    lineWidth = 0,
    autoFit = "shrinkText",
  },
) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
  shape.text = text;
  shape.text.style = {
    fontSize,
    color,
    bold,
    typeface,
    alignment,
    verticalAlignment,
    autoFit,
  };
  return shape;
}

function addRect(
  slide,
  {
    name,
    left,
    top,
    width,
    height,
    fill = COLORS.panel,
    lineFill = "none",
    lineWidth = 0,
    geometry = "rect",
  },
) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addRule(slide, name, left, top, width, fill = COLORS.faint, weight = 1) {
  return slide.shapes.add({
    geometry: "straightConnector1",
    name,
    position: { left, top, width, height: 0 },
    fill: "none",
    line: { style: "solid", fill, width: weight },
  });
}

function addFooter(slide, pageNumber, section) {
  addText(slide, {
    name: `footer-section-${pageNumber}`,
    text: section,
    left: 64,
    top: 680,
    width: 520,
    height: 22,
    fontSize: 13,
    color: COLORS.muted,
    typeface: FONTS.cn,
  });
  addText(slide, {
    name: `footer-page-${pageNumber}`,
    text: String(pageNumber),
    left: 1170,
    top: 680,
    width: 46,
    height: 22,
    fontSize: 13,
    color: COLORS.muted,
    alignment: "right",
    typeface: FONTS.latin,
  });
}

function addSlideTitle(slide, title, pageNumber, section, accent = COLORS.blue) {
  addText(slide, {
    name: `slide-title-${pageNumber}`,
    text: title,
    left: 64,
    top: 42,
    width: 1120,
    height: 66,
    fontSize: 36,
    bold: true,
    color: COLORS.ink,
    typeface: FONTS.cn,
  });
  addRule(slide, `title-rule-${pageNumber}`, 64, 116, 1152, accent, 3);
  addFooter(slide, pageNumber, section);
}

async function addFigure(slide, filename, alt, position, fit = "contain") {
  const bytes = await fs.readFile(path.join(FIGURE_DIR, filename));
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit,
    position,
  });
}

async function writeBlob(targetPath, blob) {
  await fs.writeFile(targetPath, new Uint8Array(await blob.arrayBuffer()));
}

function addEvidenceTag(slide, text, left, top, width, fill, color) {
  addRect(slide, {
    name: `tag-bg-${text}`,
    left,
    top,
    width,
    height: 34,
    fill,
    geometry: "roundRect",
  });
  addText(slide, {
    name: `tag-text-${text}`,
    text,
    left: left + 10,
    top: top + 4,
    width: width - 20,
    height: 26,
    fontSize: 16,
    bold: true,
    color,
    alignment: "center",
    verticalAlignment: "middle",
  });
}

async function buildDeck() {
  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  // 1. Cover: restrained title + editable primary metric.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addRect(slide, {
      name: "cover-accent-edge",
      left: 0,
      top: 0,
      width: 16,
      height: 720,
      fill: COLORS.blue,
    });
    addText(slide, {
      name: "cover-kicker",
      text: "SEISMOFLUX · 冻结历史回放",
      left: 68,
      top: 66,
      width: 520,
      height: 32,
      fontSize: 17,
      bold: true,
      color: COLORS.blue,
      typeface: FONTS.cn,
    });
    addText(slide, {
      name: "cover-title",
      text: "相同面积上限下，多命中4个规则化震群",
      left: 68,
      top: 150,
      width: 1138,
      height: 90,
      fontSize: 54,
      bold: true,
      color: COLORS.ink,
    });
    addText(slide, {
      name: "cover-subtitle",
      text: "将起报前30天的近期地震活动叠加到长期背景",
      left: 70,
      top: 272,
      width: 850,
      height: 52,
      fontSize: 28,
      color: COLORS.muted,
    });
    addRect(slide, {
      name: "cover-metric-field",
      left: 0,
      top: 390,
      width: 1280,
      height: 230,
      fill: COLORS.blueLight,
    });
    addText(slide, {
      name: "cover-before-label",
      text: "长期背景 B0",
      left: 112,
      top: 434,
      width: 230,
      height: 34,
      fontSize: 20,
      bold: true,
      color: COLORS.muted,
    });
    addText(slide, {
      name: "cover-metric-before",
      text: "5/21",
      left: 112,
      top: 478,
      width: 210,
      height: 90,
      fontSize: 64,
      bold: true,
      color: COLORS.muted,
      typeface: FONTS.latin,
    });
    addText(slide, {
      name: "cover-metric-arrow",
      text: "→",
      left: 430,
      top: 482,
      width: 130,
      height: 72,
      fontSize: 56,
      bold: true,
      color: COLORS.blue,
      alignment: "center",
    });
    addText(slide, {
      name: "cover-metric-after",
      text: "9/21",
      left: 712,
      top: 478,
      width: 220,
      height: 90,
      fontSize: 64,
      bold: true,
      color: COLORS.green,
      typeface: FONTS.latin,
    });
    addText(slide, {
      name: "cover-after-label",
      text: "长期背景 + 近期30天 B0_R30",
      left: 712,
      top: 434,
      width: 420,
      height: 34,
      fontSize: 20,
      bold: true,
      color: COLORS.green,
    });
    addText(slide, {
      name: "cover-boundary",
      text: "30天预测窗 · 600,000 km²面积上限 · 21个规则化 M5–6 震群 · 真实前瞻为0期",
      left: 72,
      top: 652,
      width: 1136,
      height: 36,
      fontSize: 18,
      color: COLORS.orange,
      bold: true,
      alignment: "center",
    });
    addNotes(slide, [
      SOURCE_PATHS.metrics,
      SOURCE_PATHS.figureCaptions,
      SOURCE_PATHS.d1Acceptance,
    ]);
  }

  // 2. Scientific question.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(
      slide,
      "科学问题只有一个：有限报警面积里，能多覆盖几个规则化震群？",
      2,
      "问题与评价",
    );
    addText(slide, {
      name: "question-area-number",
      text: "600,000 km²上限",
      left: 68,
      top: 180,
      width: 520,
      height: 84,
      fontSize: 56,
      bold: true,
      color: COLORS.blue,
      typeface: FONTS.latin,
    });
    addText(slide, {
      name: "question-area-label",
      text: "两种方法使用同样的面积上限",
      left: 70,
      top: 278,
      width: 500,
      height: 48,
      fontSize: 24,
      bold: true,
    });
    addText(slide, {
      name: "question-area-body",
      text: "这样比较的是排序能力，而不是谁把更多地方都涂成高风险。",
      left: 70,
      top: 340,
      width: 500,
      height: 92,
      fontSize: 22,
      color: COLORS.muted,
    });
    addRect(slide, {
      name: "question-divider-vertical",
      left: 638,
      top: 170,
      width: 2,
      height: 390,
      fill: COLORS.faint,
    });
    addText(slide, {
      name: "question-clusters-number",
      text: "21 个",
      left: 706,
      top: 180,
      width: 300,
      height: 84,
      fontSize: 56,
      bold: true,
      color: COLORS.green,
      typeface: FONTS.cn,
    });
    addText(slide, {
      name: "question-clusters-label",
      text: "规则化 M5–6 震群",
      left: 708,
      top: 278,
      width: 470,
      height: 48,
      fontSize: 24,
      bold: true,
    });
    addText(slide, {
      name: "question-clusters-body",
      text: "把同一余震序列合并，避免一次活跃序列被重复计成很多次命中。",
      left: 708,
      top: 340,
      width: 470,
      height: 92,
      fontSize: 22,
      color: COLORS.muted,
    });
    addRect(slide, {
      name: "question-takeaway-bg",
      left: 70,
      top: 500,
      width: 1108,
      height: 90,
      fill: COLORS.panel,
    });
    addText(slide, {
      name: "question-takeaway",
      text: "评价标准：在相同面积上限下，看未来30天有多少规则化震群落入报警区。",
      left: 100,
      top: 522,
      width: 1048,
      height: 46,
      fontSize: 26,
      bold: true,
      alignment: "center",
    });
    addNotes(slide, [SOURCE_PATHS.clusters, SOURCE_PATHS.d1Acceptance]);
  }

  // 3. Data boundary.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "最终结果主要由地震目录驱动，并非所有数据都进入模型", 3, "数据边界");
    addRect(slide, {
      name: "data-column-used-accent",
      left: 64,
      top: 152,
      width: 346,
      height: 10,
      fill: COLORS.green,
    });
    addRect(slide, {
      name: "data-column-studied-accent",
      left: 467,
      top: 152,
      width: 346,
      height: 10,
      fill: COLORS.orange,
    });
    addRect(slide, {
      name: "data-column-excluded-accent",
      left: 870,
      top: 152,
      width: 346,
      height: 10,
      fill: COLORS.red,
    });
    addText(slide, {
      name: "data-used-heading",
      text: "进入最终 B0_R30",
      left: 64,
      top: 184,
      width: 346,
      height: 48,
      fontSize: 26,
      bold: true,
      color: COLORS.green,
    });
    addText(slide, {
      name: "data-used-body",
      text:
        "40,898 条历史地震目录\n\n约25 km网格，共15,697格\n\n长期75 km平滑背景 + 起报前30天近期活动\n\n39个目标无关构造区用于异常空间置乱和区域稳健性",
      left: 64,
      top: 252,
      width: 346,
      height: 330,
      fontSize: 20,
      color: COLORS.ink,
    });
    addText(slide, {
      name: "data-studied-heading",
      text: "研究过，但未带来主指标提升",
      left: 467,
      top: 184,
      width: 346,
      height: 64,
      fontSize: 26,
      bold: true,
      color: COLORS.orange,
    });
    addText(slide, {
      name: "data-studied-body",
      text:
        "异常表：205个报告期、3,217,885条特征\n\n异常模型做过时间与空间置乱，但没有形成主终点增益\n\n断层与底图未进入最终 B0_R30 排序",
      left: 467,
      top: 270,
      width: 346,
      height: 280,
      fontSize: 20,
      color: COLORS.ink,
    });
    addText(slide, {
      name: "data-excluded-heading",
      text: "明确排除",
      left: 870,
      top: 184,
      width: 346,
      height: 48,
      fontSize: 26,
      bold: true,
      color: COLORS.red,
    });
    addText(slide, {
      name: "data-excluded-body",
      text:
        "人工预测地点、震级与时间\n\n起报日之后才出现的地震或异常信息\n\n真实震中生成的候选区、边界或空间加密位置",
      left: 870,
      top: 252,
      width: 346,
      height: 250,
      fontSize: 20,
      color: COLORS.ink,
    });
    addText(slide, {
      name: "data-boundary-note",
      text: "结论：异常数据不是被忽略，而是经过检验后没有进入冻结前瞻模型。",
      left: 64,
      top: 610,
      width: 1152,
      height: 44,
      fontSize: 22,
      bold: true,
      color: COLORS.muted,
      alignment: "center",
    });
    addNotes(slide, [
      SOURCE_PATHS.sourceBoundary,
      SOURCE_PATHS.d1Acceptance,
      SOURCE_PATHS.p1Protocol,
    ]);
  }

  // 4. Method overview.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "模型只看起报前信息：长期背景管“常发”，近期目录管“正在活跃”", 4, "冻结方法");
    await addFigure(
      slide,
      "figure_05_method_overview.png",
      "从起报前可见地震目录到固定面积报警与规则化震群评价的流程",
      { left: 70, top: 145, width: 1140, height: 475 },
      "contain",
    );
    addText(slide, {
      name: "method-bottom-note",
      text: "B0_R30 是相对强度排序，不是某地未来30天的绝对发震概率。",
      left: 128,
      top: 622,
      width: 1024,
      height: 38,
      fontSize: 20,
      bold: true,
      color: COLORS.orange,
      alignment: "center",
    });
    addNotes(slide, [
      `${SOURCE_PATHS.figureCaptions}（Figure 05）`,
      SOURCE_PATHS.p1Protocol,
    ]);
  }

  // 5. Primary result.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "主结果：相同600,000 km²上限下，召回从5/21提高到9/21", 5, "总体效果", COLORS.green);
    addEvidenceTag(slide, "+4 个规则化震群", 72, 132, 208, COLORS.greenLight, COLORS.green);
    addEvidenceTag(slide, "+19.05 个百分点", 298, 132, 216, COLORS.blueLight, COLORS.blue);
    await addFigure(
      slide,
      "figure_01_primary_result.png",
      "固定报警面积下B0与B0_R30的召回、面积曲线、分折与证据边界",
      { left: 62, top: 174, width: 1156, height: 474 },
      "contain",
    );
    addNotes(slide, [
      `${SOURCE_PATHS.figureCaptions}（Figure 01）`,
      SOURCE_PATHS.metrics,
      SOURCE_PATHS.d1Acceptance,
    ]);
  }

  // 6. All clusters.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "看完全部21群：新增4群，没有出现“原来命中、后来丢失”", 6, "全部目标", COLORS.green);
    await addFigure(
      slide,
      "figure_02_all_cluster_outcomes.png",
      "全部21个规则化M5至6震群在B0和B0_R30下的命中状态",
      { left: 58, top: 138, width: 1164, height: 505 },
      "contain",
    );
    addNotes(slide, [
      `${SOURCE_PATHS.figureCaptions}（Figure 02）`,
      SOURCE_PATHS.clusters,
    ]);
  }

  // 7. Folds and uncertainty: editable native chart.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "提升集中在前两段时间外推，第三段没有改善", 7, "稳健性与边界", COLORS.orange);
    slide.charts.add("bar", {
      position: { left: 68, top: 172, width: 620, height: 420 },
      categories: ["折1", "折2", "折3"],
      series: [
        { name: "B0", values: [3, 2, 0], fill: "#A9B5BE" },
        { name: "B0_R30", values: [5, 4, 0], fill: COLORS.green },
      ],
      hasLegend: true,
      legend: { position: "bottom", overlay: false },
      dataLabels: { showValue: true, position: "outEnd" },
      chartFill: COLORS.canvas,
      chartLine: { style: "solid", width: 0, fill: COLORS.canvas },
      plotAreaFill: { type: "none" },
      plotAreaLine: { style: "solid", width: 0, fill: COLORS.canvas },
      xAxis: {
        visible: true,
        deleted: false,
        line: { style: "solid", width: 1, fill: COLORS.faint },
        textStyle: { typeface: FONTS.cn, fontSize: 18, color: COLORS.ink },
      },
      yAxis: {
        visible: true,
        deleted: false,
        min: 0,
        max: 6,
        majorUnit: 1,
        majorGridlines: { style: "solid", width: 1, fill: COLORS.faint },
        line: { style: "solid", width: 0, fill: COLORS.canvas },
        textStyle: { typeface: FONTS.latin, fontSize: 15, color: COLORS.muted },
      },
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 85 },
    });
    addText(slide, {
      name: "uncertainty-gain",
      text: "+19.05 pp",
      left: 760,
      top: 178,
      width: 390,
      height: 76,
      fontSize: 48,
      bold: true,
      color: COLORS.green,
      typeface: FONTS.cn,
    });
    addText(slide, {
      name: "uncertainty-ci",
      text: "2000次配对震群 bootstrap\n95%区间：+4.76 ～ +38.10 pp\n正增益复本比例 = 0.9905\n（非模型正确概率）",
      left: 760,
      top: 268,
      width: 410,
      height: 150,
      fontSize: 22,
      color: COLORS.ink,
    });
    addRect(slide, {
      name: "uncertainty-warning-bg",
      left: 738,
      top: 448,
      width: 446,
      height: 126,
      fill: COLORS.orangeLight,
    });
    addText(slide, {
      name: "uncertainty-warning",
      text: "需要正视的边界\n折3为 0/7 → 0/7，当前提升还不能视为跨时期、跨区域稳定成立。",
      left: 766,
      top: 468,
      width: 390,
      height: 88,
      fontSize: 20,
      bold: true,
      color: COLORS.orange,
    });
    addNotes(slide, [SOURCE_PATHS.metrics, SOURCE_PATHS.d1Acceptance]);
  }

  // 8. Two detailed cases.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "两个直观震例：近期活动把长期背景中不突出的区域推入报警区", 8, "震例解释", COLORS.orange);
    await addFigure(
      slide,
      "figure_03_case_studies.png",
      "库车M5.5与肃北M5.0的长期背景、近期活动混合结果和起报时间线",
      { left: 54, top: 132, width: 1172, height: 520 },
      "contain",
    );
    addNotes(slide, [
      `${SOURCE_PATHS.figureCaptions}（Figure 03）`,
      SOURCE_PATHS.cases,
      SOURCE_PATHS.payload,
    ]);
  }

  // 9. All four rank shifts.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "四个新增命中都来自优先级的大幅前移", 9, "震例解释", COLORS.green);
    await addFigure(
      slide,
      "figure_04_case_rank_shifts.png",
      "四个新增命中震群在B0与B0_R30中的目标格排名变化",
      { left: 60, top: 145, width: 780, height: 470 },
      "contain",
    );
    addText(slide, {
      name: "rank-heading",
      text: "目标格名次（越小越优先）",
      left: 876,
      top: 164,
      width: 320,
      height: 44,
      fontSize: 23,
      bold: true,
      color: COLORS.blue,
    });
    addText(slide, {
      name: "rank-cases",
      text:
        "肃北 M5.0\n3912 → 194\n\n杂多 M5.5\n1669 → 153\n\n库车 M5.0\n2146 → 3\n\n库车 M5.5\n1725 → 3",
      left: 876,
      top: 230,
      width: 300,
      height: 330,
      fontSize: 22,
      color: COLORS.ink,
      typeface: FONTS.cn,
    });
    addText(slide, {
      name: "rank-caveat",
      text: "两个库车事件落在同一个约25 km网格；震例用于解释，主结论仍以全部21群为准。",
      left: 868,
      top: 575,
      width: 320,
      height: 70,
      fontSize: 17,
      color: COLORS.muted,
    });
    addNotes(slide, [
      `${SOURCE_PATHS.figureCaptions}（Figure 04）`,
      SOURCE_PATHS.cases,
    ]);
  }

  // 10. What the evidence does and does not show.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "当前证据支持“值得前瞻验证”，还不支持“已经实现真实预测”", 10, "科学结论", COLORS.orange);
    addText(slide, {
      name: "supported-heading",
      text: "已经得到的证据",
      left: 72,
      top: 170,
      width: 500,
      height: 48,
      fontSize: 28,
      bold: true,
      color: COLORS.green,
    });
    addText(slide, {
      name: "supported-body",
      text:
        "• 固定面积历史回放中，规则化震群召回增加4群\n\n• 21群配对结果为4个新增命中、0个反向丢失\n\n• 条件Bootstrap区间为+4.76至+38.10个百分点",
      left: 72,
      top: 248,
      width: 500,
      height: 278,
      fontSize: 22,
      color: COLORS.ink,
    });
    addRect(slide, {
      name: "conclusion-divider",
      left: 635,
      top: 164,
      width: 2,
      height: 430,
      fill: COLORS.faint,
    });
    addText(slide, {
      name: "unsupported-heading",
      text: "仍然不能下的结论",
      left: 700,
      top: 170,
      width: 500,
      height: 48,
      fontSize: 28,
      bold: true,
      color: COLORS.orange,
    });
    addText(slide, {
      name: "unsupported-body",
      text:
        "• 截至2026年8月31日真实前瞻为0期\n\n• 折3没有改善，地理与时期泛化仍待验证\n\n• 相对强度不是绝对发震概率\n\n• ETAS数值资格未通过，当前无法形成可评价对照分数",
      left: 700,
      top: 248,
      width: 500,
      height: 310,
      fontSize: 22,
      color: COLORS.ink,
    });
    addText(slide, {
      name: "conclusion-bottom",
      text: "因此，最有科学价值的下一步是保持模型不变，接受真实前瞻检验。",
      left: 136,
      top: 610,
      width: 1008,
      height: 42,
      fontSize: 24,
      bold: true,
      color: COLORS.blue,
      alignment: "center",
    });
    addNotes(slide, [
      SOURCE_PATHS.d1Acceptance,
      SOURCE_PATHS.etasQualification,
      SOURCE_PATHS.p1Protocol,
    ]);
  }

  // 11. Prospective timeline. Connector first, then milestones.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addSlideTitle(slide, "下一步保持模型冻结，从首个合法时刻开始真实前瞻", 11, "真实前瞻", COLORS.blue);
    addText(slide, {
      name: "prospective-intro",
      text: "冻结同一模型、同一面积上限、同一指标；预测保存后不得回写。",
      left: 70,
      top: 150,
      width: 1100,
      height: 50,
      fontSize: 26,
      bold: true,
      color: COLORS.ink,
      alignment: "center",
    });
    slide.shapes.add({
      geometry: "straightConnector1",
      name: "prospective-timeline-line",
      position: { left: 170, top: 350, width: 940, height: 0 },
      fill: "none",
      line: { style: "solid", fill: COLORS.blue, width: 3 },
    });
    const milestones = [
      {
        x: 220,
        label: "首期合法起报",
        date: "2026-09-10 00:00",
        body: "保存不可回写的空间排序、静态图和离线交互页",
        fill: COLORS.blue,
      },
      {
        x: 570,
        label: "30天成熟",
        date: "主终点评价",
        body: "按相同报警面积上限比较规则化 M5–6 震群召回",
        fill: COLORS.green,
      },
      {
        x: 920,
        label: "90天成熟",
        date: "补充评价",
        body: "检验较长窗口下结论是否一致",
        fill: COLORS.orange,
      },
    ];
    for (const [index, item] of milestones.entries()) {
      addRect(slide, {
        name: `prospective-node-${index + 1}`,
        left: item.x,
        top: 330,
        width: 40,
        height: 40,
        fill: item.fill,
        geometry: "ellipse",
      });
      addText(slide, {
        name: `prospective-label-${index + 1}`,
        text: item.label,
        left: item.x - 70,
        top: 248,
        width: 180,
        height: 40,
        fontSize: 26,
        bold: true,
        color: item.fill,
        alignment: "center",
      });
      addText(slide, {
        name: `prospective-date-${index + 1}`,
        text: item.date,
        left: item.x - 82,
        top: 292,
        width: 204,
        height: 30,
        fontSize: 17,
        color: COLORS.muted,
        alignment: "center",
      });
      addText(slide, {
        name: `prospective-body-${index + 1}`,
        text: item.body,
        left: item.x - 110,
        top: 405,
        width: 260,
        height: 110,
        fontSize: 20,
        color: COLORS.ink,
        alignment: "center",
      });
    }
    addRect(slide, {
      name: "prospective-status-bg",
      left: 236,
      top: 566,
      width: 808,
      height: 66,
      fill: COLORS.orangeLight,
    });
    addText(slide, {
      name: "prospective-status",
      text: "当前状态：真实前瞻已获授权，但首期起报尚未发生，目前为0期。",
      left: 262,
      top: 584,
      width: 756,
      height: 34,
      fontSize: 22,
      bold: true,
      color: COLORS.orange,
      alignment: "center",
    });
    addNotes(slide, [SOURCE_PATHS.p1Protocol, SOURCE_PATHS.p1Authorization]);
  }

  // 12. Close on the scientific decision.
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.canvas;
    addText(slide, {
      name: "close-kicker",
      text: "SEISMOFLUX · 当前科学结论",
      left: 72,
      top: 70,
      width: 560,
      height: 34,
      fontSize: 18,
      bold: true,
      color: COLORS.blue,
      typeface: FONTS.cn,
    });
    addText(slide, {
      name: "close-title",
      text: "历史回放给出积极信号，\n真实前瞻决定它能否成立",
      left: 72,
      top: 150,
      width: 810,
      height: 150,
      fontSize: 54,
      bold: true,
      color: COLORS.ink,
    });
    addRect(slide, {
      name: "close-summary-bg",
      left: 72,
      top: 372,
      width: 1136,
      height: 190,
      fill: COLORS.panel,
    });
    addText(slide, {
      name: "close-summary-1",
      text: "+4",
      left: 110,
      top: 408,
      width: 160,
      height: 72,
      fontSize: 52,
      bold: true,
      color: COLORS.green,
      typeface: FONTS.latin,
      alignment: "center",
    });
    addText(slide, {
      name: "close-summary-1-label",
      text: "历史回放新增命中震群",
      left: 92,
      top: 495,
      width: 196,
      height: 36,
      fontSize: 18,
      color: COLORS.muted,
      alignment: "center",
    });
    addText(slide, {
      name: "close-summary-2",
      text: "折3：0 → 0",
      left: 408,
      top: 420,
      width: 320,
      height: 54,
      fontSize: 34,
      bold: true,
      color: COLORS.orange,
      typeface: FONTS.latin,
      alignment: "center",
    });
    addText(slide, {
      name: "close-summary-2-label",
      text: "泛化边界仍然存在",
      left: 438,
      top: 495,
      width: 260,
      height: 36,
      fontSize: 18,
      color: COLORS.muted,
      alignment: "center",
    });
    addText(slide, {
      name: "close-summary-3",
      text: "30 / 90 天",
      left: 846,
      top: 420,
      width: 290,
      height: 54,
      fontSize: 34,
      bold: true,
      color: COLORS.blue,
      typeface: FONTS.latin,
      alignment: "center",
    });
    addText(slide, {
      name: "close-summary-3-label",
      text: "从0期开始真实前瞻",
      left: 860,
      top: 495,
      width: 260,
      height: 36,
      fontSize: 18,
      color: COLORS.muted,
      alignment: "center",
    });
    addText(slide, {
      name: "close-final-line",
      text: "保留这次提升最可信的方式，是让下一批真实地震来回答。",
      left: 168,
      top: 612,
      width: 944,
      height: 50,
      fontSize: 27,
      bold: true,
      color: COLORS.blue,
      alignment: "center",
    });
    addNotes(slide, [
      SOURCE_PATHS.metrics,
      SOURCE_PATHS.d1Acceptance,
      SOURCE_PATHS.p1Protocol,
    ]);
  }

  return presentation;
}

async function main() {
  const outputDir = process.argv[2]
    ? path.resolve(process.argv[2])
    : DEFAULT_OUTPUT_DIR;
  const finalPptx = process.argv[3]
    ? path.resolve(process.argv[3])
    : path.join(outputDir, "seismoflux_science_presentation_zh.pptx");
  const qaDir = path.join(outputDir, "artifact_tool_qa");

  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const presentation = await buildDeck();
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(qaDir, `${stem}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(qaDir, `${stem}.layout.json`), await layout.text());
  }

  const montage = await presentation.export({
    format: "webp",
    montage: true,
    scale: 1,
  });
  await writeBlob(path.join(qaDir, "deck-montage.webp"), montage);
  await fs.writeFile(
    path.join(qaDir, "source-notes.txt"),
    Object.values(SOURCE_PATHS).join("\n") + "\n",
    "utf8",
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  process.stdout.write(`${finalPptx}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
