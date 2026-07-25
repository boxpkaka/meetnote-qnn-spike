#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const outputPath =
  process.argv[2] ??
  path.join(root, "build/qnn-spike/qnn-meeting-calibration-128.txt");
const inputPaths = [
  path.join(
    root,
    "build/qnn-spike/meeting-summary-stress-20260719/inputs/long-96.json",
  ),
  path.join(
    root,
    "build/qnn-spike/meeting-summary-stress-20260719/inputs/extreme-320.json",
  ),
];

const meetings = inputPaths.map((inputPath) =>
  JSON.parse(fs.readFileSync(inputPath, "utf8")),
);
const transcriptTasks = [
  '只依据以下会议原文提取明确决议、行动项和风险，输出紧凑合法 JSON，所有 evidence_ids 必须原样复制。',
  '阅读以下会议片段，区分 confirmed、proposed 和 unclear，不得补充原文没有的负责人或日期。',
  '从以下中文会议转写中提取数字、百分比、日期、金额和 utterance id，字符必须逐字一致。',
  '将以下会议片段整理为 headline、overview、topics 和 evidence_ids，不要 Markdown 或解释。',
];

const transcriptUtterances = [
  ...meetings[0].utterances.slice(18),
  ...meetings[1].utterances.slice(96),
];
const transcriptSamples = [];
for (let index = 0; index + 1 < transcriptUtterances.length; index += 2) {
  const pair = transcriptUtterances.slice(index, index + 2);
  const body = pair
    .map(
      (utterance) =>
        `[id=${utterance.id}][speaker=${utterance.speaker_name}/${utterance.role}] ${utterance.text}`,
    )
    .join(" ");
  transcriptSamples.push(
    `${transcriptTasks[transcriptSamples.length % transcriptTasks.length]} transcript: ${body}`,
  );
  if (transcriptSamples.length === 80) {
    break;
  }
}

const numberSamples = Array.from({ length: 24 }, (_, index) => {
  const n = index + 1;
  const percent = 73 + (index % 25);
  const month = (index % 12) + 1;
  const day = (index % 27) + 1;
  const amount = (index + 3) * 1700;
  return [
    "精确复制输入中的数字、日期、金额和证据编号，并输出合法 JSON；不要换算、改写或猜测。",
    `输入：[id=utt-${n * 3}] 完成率为百分之${percent}，计划在${month}月${day}日前完成，预算${amount}元。`,
    `输出格式：{"accuracy":"百分之${percent}","due":"${month}月${day}日","budget":"${amount}元","evidence_ids":["utt-${n * 3}"]}`,
  ].join(" ");
});

const schemaSamples = Array.from({ length: 16 }, (_, index) => {
  const id = 400 + index;
  const status = index % 3 === 0 ? "confirmed" : index % 3 === 1 ? "proposed" : "unclear";
  const priority = index % 2 === 0 ? "high" : "unknown";
  return [
    "会议纪要 JSON 字段合同训练样本。",
    `原文：[id=utt-${id}] ${status === "confirmed" ? "会议确认" : "会上提议"}由负责人S${(index % 8) + 1}在下周完成离线验收，未说明具体日期。`,
    `目标结构：{"decisions":[{"decision":"离线验收","owner":"S${(index % 8) + 1}","status":"${status}","evidence_ids":["utt-${id}"]}],"action_items":[{"owner":"S${(index % 8) + 1}","task":"完成离线验收","due":"待确认","priority":"${priority}","evidence_ids":["utt-${id}"]}]}`,
  ].join(" ");
});

const contractSamples = [
  "只输出合法 JSON 对象。顶层字段顺序为 schema_version、meeting_title、headline、overview、topics、decisions、action_items、risks、open_questions、follow_up_message。",
  "evidence_ids 只能引用输入中存在的 utterance id，例如 utt-0、utt-1、utt-17；连字符和数字都必须原样保留。",
  "没有明确决议时 decisions 输出 []；没有明确行动项时 action_items 输出 []；没有真实风险时 risks 输出 []。",
  "不得把提议写成确认决议。status 只能是 confirmed、proposed、unclear 之一。",
  "原文没有负责人、截止时间或缓解措施时写待确认，不得依据常识补全。",
  "会议数字需要逐字复制：百分之九十七不能改成百分之十七，十二万元不能改成二万元。",
  "输出必须使用英文半角 JSON 引号、冒号和逗号，不要代码块、Markdown、思考过程或额外解释。",
  "headline 和 overview 使用中文；schema key、枚举值和 utterance id 保持合同规定的英文形式。",
];

const samples = [
  ...transcriptSamples,
  ...numberSamples,
  ...schemaSamples,
  ...contractSamples,
].map((sample) => sample.replace(/\s+/g, " ").trim());

if (samples.length !== 128) {
  throw new Error(`expected 128 samples, got ${samples.length}`);
}
if (new Set(samples).size !== samples.length) {
  throw new Error("calibration samples must be unique");
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${samples.join("\n")}\n`, "utf8");
console.log(
  JSON.stringify(
    {
      outputPath,
      samples: samples.length,
      transcriptSamples: transcriptSamples.length,
      numberSamples: numberSamples.length,
      schemaSamples: schemaSamples.length,
      contractSamples: contractSamples.length,
      minChars: Math.min(...samples.map((sample) => sample.length)),
      maxChars: Math.max(...samples.map((sample) => sample.length)),
    },
    null,
    2,
  ),
);
