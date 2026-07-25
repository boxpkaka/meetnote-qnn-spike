#!/usr/bin/env node

import fs from "node:fs";

function readTrace(prefix) {
  const records = fs
    .readFileSync(`${prefix}.layers.jsonl`, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  const values = fs.readFileSync(`${prefix}.layers.f32`);
  return {
    records: new Map(
      records.map((record) => [`${record.kind}:${record.layer}`, record]),
    ),
    values,
  };
}

function readValues(trace, key) {
  const record = trace.records.get(key);
  if (!record) {
    throw new Error(`missing trace record: ${key}`);
  }
  const values = new Float32Array(record.elements);
  for (let index = 0; index < record.elements; index += 1) {
    values[index] = trace.values.readFloatLE(
      (record.offset_floats + index) * 4,
    );
  }
  return { record, values };
}

function cosine(left, right) {
  if (left.length !== right.length) {
    throw new Error(`element count differs: ${left.length} vs ${right.length}`);
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] ** 2;
    rightNorm += right[index] ** 2;
  }
  return dot / Math.sqrt(leftNorm * rightNorm);
}

function wrapAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function phaseAngles(qNorm, attnQ) {
  const headDim = qNorm.record.shape.at(-1);
  if (
    !Number.isInteger(headDim) ||
    headDim % 2 !== 0 ||
    qNorm.values.length !== attnQ.values.length ||
    qNorm.values.length % headDim !== 0
  ) {
    throw new Error("layer 0 Q tensors do not have a compatible RoPE shape");
  }

  const heads = qNorm.values.length / headDim;
  const half = headDim / 2;
  const angles = [];
  for (let dimension = 0; dimension < half; dimension += 1) {
    let sinSum = 0;
    let cosSum = 0;
    let samples = 0;
    for (let head = 0; head < heads; head += 1) {
      const offset = head * headDim;
      const first = qNorm.values[offset + dimension];
      const second = qNorm.values[offset + dimension + half];
      const rotatedFirst = attnQ.values[offset + dimension];
      const rotatedSecond = attnQ.values[offset + dimension + half];
      const denominator = first ** 2 + second ** 2;
      if (denominator <= 1e-8) {
        continue;
      }
      const cosValue =
        (first * rotatedFirst + second * rotatedSecond) / denominator;
      const sinValue =
        (first * rotatedSecond - second * rotatedFirst) / denominator;
      const angle = Math.atan2(sinValue, cosValue);
      sinSum += Math.sin(angle);
      cosSum += Math.cos(angle);
      samples += 1;
    }
    if (samples === 0) {
      throw new Error(`no usable phase samples for dimension ${dimension}`);
    }
    angles.push(Math.atan2(sinSum, cosSum));
  }
  return angles;
}

const args = process.argv.slice(2);
const referencePrefix = args.shift();
const candidatePrefix = args.shift();
if (!referencePrefix || !candidatePrefix) {
  console.error(
    "Usage: check_rope_trace_parity.mjs <reference-prefix> <candidate-prefix> [min-attn-q-cosine=0.99] " +
      "[--candidate-qnorm-prefix <prefix>] [--candidate-qnorm-key <kind:layer>] " +
      "[--candidate-attn-key <kind:layer>]",
  );
  process.exit(2);
}
let thresholdValue = "0.99";
if (args[0] && !args[0].startsWith("--")) {
  thresholdValue = args.shift();
}
const options = {
  candidateQNormPrefix: candidatePrefix,
  candidateQNormKey: "q_norm:0",
  candidateAttnKey: "attn_q:0",
};
while (args.length > 0) {
  const flag = args.shift();
  const value = args.shift();
  if (!value) {
    throw new Error(`missing value for ${flag}`);
  }
  if (flag === "--candidate-qnorm-prefix") {
    options.candidateQNormPrefix = value;
  } else if (flag === "--candidate-qnorm-key") {
    options.candidateQNormKey = value;
  } else if (flag === "--candidate-attn-key") {
    options.candidateAttnKey = value;
  } else {
    throw new Error(`unknown option: ${flag}`);
  }
}
const threshold = Number(thresholdValue);
if (!Number.isFinite(threshold) || threshold < -1 || threshold > 1) {
  throw new Error(`invalid cosine threshold: ${thresholdValue}`);
}

const reference = readTrace(referencePrefix);
const candidate = readTrace(candidatePrefix);
const candidateQNormTrace =
  options.candidateQNormPrefix === candidatePrefix
    ? candidate
    : readTrace(options.candidateQNormPrefix);
const referenceQNorm = readValues(reference, "q_norm:0");
const candidateQNorm = readValues(
  candidateQNormTrace,
  options.candidateQNormKey,
);
const referenceAttnQ = readValues(reference, "attn_q:0");
const candidateAttnQ = readValues(candidate, options.candidateAttnKey);
const referenceAngles = phaseAngles(referenceQNorm, referenceAttnQ);
const candidateAngles = phaseAngles(candidateQNorm, candidateAttnQ);
const angleErrors = referenceAngles.map((angle, index) => ({
  dimension: index,
  reference_angle: angle,
  candidate_angle: candidateAngles[index],
  absolute_error: Math.abs(wrapAngle(candidateAngles[index] - angle)),
}));
angleErrors.sort((left, right) => right.absolute_error - left.absolute_error);

const result = {
  q_norm_cosine: cosine(referenceQNorm.values, candidateQNorm.values),
  attn_q_cosine: cosine(referenceAttnQ.values, candidateAttnQ.values),
  phase_rmse: Math.sqrt(
    angleErrors.reduce((sum, item) => sum + item.absolute_error ** 2, 0) /
      angleErrors.length,
  ),
  candidate_qnorm_key: options.candidateQNormKey,
  candidate_attn_key: options.candidateAttnKey,
  worst_phase_dimensions: angleErrors.slice(0, 8),
  min_attn_q_cosine: threshold,
};
result.passed = result.attn_q_cosine >= threshold;
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
if (!result.passed) {
  process.exitCode = 1;
}
