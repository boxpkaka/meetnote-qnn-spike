#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

function readProbe(prefix) {
  const records = fs
    .readFileSync(`${prefix}.jsonl`, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  const header = records.find((record) => record.type === "header");
  const steps = records.filter((record) => record.type === "step");
  const footer = records.find((record) => record.type === "footer");
  if (
    !header ||
    !footer ||
    footer.steps_written !== header.steps ||
    steps.length !== header.steps
  ) {
    throw new Error(`invalid probe metadata: ${prefix}`);
  }
  const logits = fs.readFileSync(`${prefix}.f32`);
  const expectedBytes = header.steps * header.vocab_size * 4;
  if (logits.length !== expectedBytes) {
    throw new Error(
      `invalid logits size for ${prefix}: expected ${expectedBytes}, got ${logits.length}`,
    );
  }
  return { header, steps, logits };
}

function valueAt(probe, step, token) {
  const offset = (step * probe.header.vocab_size + token) * 4;
  return probe.logits.readFloatLE(offset);
}

function compare(cpu, qnn) {
  if (
    cpu.header.steps !== qnn.header.steps ||
    cpu.header.vocab_size !== qnn.header.vocab_size
  ) {
    throw new Error("probe dimensions differ");
  }

  const stepResults = [];
  let cpuGlobalMin = Number.POSITIVE_INFINITY;
  let cpuGlobalMax = Number.NEGATIVE_INFINITY;
  let qnnGlobalMin = Number.POSITIVE_INFINITY;
  let qnnGlobalMax = Number.NEGATIVE_INFINITY;

  for (let step = 0; step < cpu.header.steps; step += 1) {
    let dot = 0;
    let cpuNorm = 0;
    let qnnNorm = 0;
    let squaredError = 0;
    let maxAbsoluteError = 0;
    let cpuTopToken = -1;
    let qnnTopToken = -1;
    let cpuTopLogit = Number.NEGATIVE_INFINITY;
    let qnnTopLogit = Number.NEGATIVE_INFINITY;

    for (let token = 0; token < cpu.header.vocab_size; token += 1) {
      const cpuValue = valueAt(cpu, step, token);
      const qnnValue = valueAt(qnn, step, token);
      dot += cpuValue * qnnValue;
      cpuNorm += cpuValue * cpuValue;
      qnnNorm += qnnValue * qnnValue;
      const difference = cpuValue - qnnValue;
      squaredError += difference * difference;
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(difference));
      if (cpuValue > cpuTopLogit) {
        cpuTopLogit = cpuValue;
        cpuTopToken = token;
      }
      if (qnnValue > qnnTopLogit) {
        qnnTopLogit = qnnValue;
        qnnTopToken = token;
      }
      cpuGlobalMin = Math.min(cpuGlobalMin, cpuValue);
      cpuGlobalMax = Math.max(cpuGlobalMax, cpuValue);
      qnnGlobalMin = Math.min(qnnGlobalMin, qnnValue);
      qnnGlobalMax = Math.max(qnnGlobalMax, qnnValue);
    }

    const cpuTop = new Set(cpu.steps[step].top.map((entry) => entry.token));
    const qnnTop = new Set(qnn.steps[step].top.map((entry) => entry.token));
    const topKOverlap = [...cpuTop].filter((token) => qnnTop.has(token)).length;
    stepResults.push({
      step,
      input_piece: cpu.steps[step].input_piece,
      target_piece: cpu.steps[step].target_piece,
      target_token: cpu.steps[step].target_token,
      cpu_top_token: cpuTopToken,
      qnn_top_token: qnnTopToken,
      top1_same: cpuTopToken === qnnTopToken,
      cpu_target_is_top1: cpuTopToken === cpu.steps[step].target_token,
      qnn_target_is_top1: qnnTopToken === qnn.steps[step].target_token,
      cosine: dot / Math.sqrt(cpuNorm * qnnNorm),
      rmse: Math.sqrt(squaredError / cpu.header.vocab_size),
      max_absolute_error: maxAbsoluteError,
      top_k_overlap: topKOverlap,
      cpu_max: cpuTopLogit,
      qnn_max: qnnTopLogit,
    });
  }

  let qnnGlobalMaxCount = 0;
  for (let step = 0; step < qnn.header.steps; step += 1) {
    for (let token = 0; token < qnn.header.vocab_size; token += 1) {
      if (valueAt(qnn, step, token) === qnnGlobalMax) {
        qnnGlobalMaxCount += 1;
      }
    }
  }

  const mean = (field) =>
    stepResults.reduce((sum, result) => sum + result[field], 0) /
    stepResults.length;
  const summary = {
    steps: cpu.header.steps,
    vocab_size: cpu.header.vocab_size,
    top1_agreement:
      stepResults.filter((result) => result.top1_same).length /
      stepResults.length,
    cpu_target_top1_rate:
      stepResults.filter((result) => result.cpu_target_is_top1).length /
      stepResults.length,
    qnn_target_top1_rate:
      stepResults.filter((result) => result.qnn_target_is_top1).length /
      stepResults.length,
    cosine_mean: mean("cosine"),
    cosine_min: Math.min(...stepResults.map((result) => result.cosine)),
    rmse_mean: mean("rmse"),
    top_k_overlap_mean: mean("top_k_overlap"),
    cpu_global_min: cpuGlobalMin,
    cpu_global_max: cpuGlobalMax,
    qnn_global_min: qnnGlobalMin,
    qnn_global_max: qnnGlobalMax,
    qnn_global_max_count: qnnGlobalMaxCount,
    first_top1_mismatch:
      stepResults.find((result) => !result.top1_same) ?? null,
  };
  return { summary, steps: stepResults };
}

const [cpuPrefix, qnnPrefix, outputPath] = process.argv.slice(2);
if (!cpuPrefix || !qnnPrefix) {
  console.error(
    "Usage: compare_teacher_forced_logits.mjs <cpu-prefix> <qnn-prefix> [output.json]",
  );
  process.exit(2);
}

const result = compare(readProbe(cpuPrefix), readProbe(qnnPrefix));
const serialized = `${JSON.stringify(result, null, 2)}\n`;
if (outputPath) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, serialized);
}
process.stdout.write(`${JSON.stringify(result.summary, null, 2)}\n`);
