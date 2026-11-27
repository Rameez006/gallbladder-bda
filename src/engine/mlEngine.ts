import {
  ConfusionMatrixCounts,
  FeatureContribution,
  FeatureImportanceItem,
  ModelKey,
  PatientPredictionInput,
  PatientPredictionOutput,
  PreprocessingResult,
  RocPoint,
  TrainedModelSummary,
} from '../types/medical';
import { MODEL_FEATURE_NAMES } from './preprocessing';

function sigmoid(z: number): number {
  const clamped = Math.max(-35, Math.min(35, z));
  return 1 / (1 + Math.exp(-clamped));
}

function evaluatePredictions(
  yTrue: number[],
  yProb: number[]
): {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  confusionMatrix: ConfusionMatrixCounts;
  rocCurve: RocPoint[];
} {
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const pred = yProb[i] >= 0.5 ? 1 : 0;
    if (yTrue[i] === 1 && pred === 1) tp++;
    else if (yTrue[i] === 0 && pred === 1) fp++;
    else if (yTrue[i] === 1 && pred === 0) fn++;
    else tn++;
  }

  const total = Math.max(1, yTrue.length);
  const accuracy = (tp + tn) / total;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // ROC Curve via sorted probability thresholds
  const pairs = yTrue
    .map((y, i) => ({ y, p: yProb[i] }))
    .sort((a, b) => b.p - a.p);

  const totalPos = Math.max(1, yTrue.filter((y) => y === 1).length);
  const totalNeg = Math.max(1, yTrue.filter((y) => y === 0).length);

  const rawCurve: RocPoint[] = [{ fpr: 0, tpr: 0 }];
  let cumTp = 0;
  let cumFp = 0;

  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].y === 1) cumTp++;
    else cumFp++;
    if (i === pairs.length - 1 || pairs[i].p !== pairs[i + 1].p) {
      rawCurve.push({
        fpr: cumFp / totalNeg,
        tpr: cumTp / totalPos,
      });
    }
  }

  // Trapezoidal AUC
  let rocAuc = 0;
  for (let i = 1; i < rawCurve.length; i++) {
    const dx = rawCurve[i].fpr - rawCurve[i - 1].fpr;
    const avgY = (rawCurve[i].tpr + rawCurve[i - 1].tpr) / 2;
    rocAuc += dx * avgY;
  }

  // Preserve high-resolution ROC curve (up to 150 points) for crisp rendering
  const rocCurve: RocPoint[] = [];
  const step = Math.max(1, Math.floor(rawCurve.length / 150));
  for (let i = 0; i < rawCurve.length; i += step) {
    rocCurve.push({
      fpr: Number(rawCurve[i].fpr.toFixed(4)),
      tpr: Number(rawCurve[i].tpr.toFixed(4)),
    });
  }
  const last = rawCurve[rawCurve.length - 1];
  if (rocCurve[rocCurve.length - 1].fpr !== 1 || rocCurve[rocCurve.length - 1].tpr !== 1) {
    rocCurve.push({ fpr: Number(last.fpr.toFixed(4)), tpr: Number(last.tpr.toFixed(4)) });
  }

  return {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1: Number(f1.toFixed(4)),
    rocAuc: Number(rocAuc.toFixed(4)),
    confusionMatrix: { tn, fp, fn, tp },
    rocCurve,
  };
}

function normalizeImportances(rawWeights: number[]): FeatureImportanceItem[] {
  const abs = rawWeights.map((w) => Math.abs(w));
  const sum = abs.reduce((a, b) => a + b, 0) || 1;
  return MODEL_FEATURE_NAMES.map((feature, idx) => ({
    feature,
    importancePct: Number(((abs[idx] / sum) * 100).toFixed(1)),
  })).sort((a, b) => b.importancePct - a.importancePct);
}

// 1. Logistic Regression (Adam optimizer with L2 regularization & balanced class weights)
function trainLogisticRegression(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[]
): TrainedModelSummary {
  const t0 = performance.now();
  const n = XTrain.length;
  const d = XTrain[0]?.length ?? MODEL_FEATURE_NAMES.length;
  const weights = new Array(d).fill(0);
  let bias = 0;

  const posCount = Math.max(1, yTrain.filter((y) => y === 1).length);
  const negCount = Math.max(1, n - posCount);
  const posWeight = n / (2 * posCount);
  const negWeight = n / (2 * negCount);

  const lr = 0.05;
  const l2 = 0.008;
  const epochs = 320;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const mW = new Array(d).fill(0);
  const vW = new Array(d).fill(0);
  let mB = 0;
  let vB = 0;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const gradW = new Array(d).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      let z = bias;
      const row = XTrain[i];
      for (let j = 0; j < d; j++) z += weights[j] * row[j];
      const p = sigmoid(z);
      const sampleW = yTrain[i] === 1 ? posWeight : negWeight;
      const err = (p - yTrain[i]) * sampleW;
      gradB += err;
      for (let j = 0; j < d; j++) gradW[j] += err * row[j];
    }

    const gB = gradB / n;
    mB = beta1 * mB + (1 - beta1) * gB;
    vB = beta2 * vB + (1 - beta2) * gB * gB;
    const mBHat = mB / (1 - Math.pow(beta1, epoch));
    const vBHat = vB / (1 - Math.pow(beta2, epoch));
    bias -= (lr * mBHat) / (Math.sqrt(vBHat) + eps);

    for (let j = 0; j < d; j++) {
      const g = gradW[j] / n + l2 * weights[j];
      mW[j] = beta1 * mW[j] + (1 - beta1) * g;
      vW[j] = beta2 * vW[j] + (1 - beta2) * g * g;
      const mHat = mW[j] / (1 - Math.pow(beta1, epoch));
      const vHat = vW[j] / (1 - Math.pow(beta2, epoch));
      weights[j] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  }

  const predictProb = (_raw: number[], scaled: number[]) => {
    let z = bias;
    for (let j = 0; j < d; j++) z += weights[j] * scaled[j];
    return sigmoid(z);
  };

  const yProbTest = XTest.map((row) => predictProb(row, row));
  const evalRes = evaluatePredictions(yTest, yProbTest);

  return {
    key: 'logistic_regression',
    name: 'Logistic Regression',
    shortName: 'LogReg',
    ...evalRes,
    trainTimeMs: Number((performance.now() - t0).toFixed(1)),
    featureImportances: normalizeImportances(weights),
    predictProb,
  };
}

// 2. CART Decision Tree (Gini impurity with balanced sample weights)
interface TreeNode {
  isLeaf: boolean;
  prob: number;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

function buildDecisionTree(
  X: number[][],
  y: number[],
  indices: number[],
  sampleWeights: number[],
  maxDepth: number,
  minSamplesLeaf: number,
  featureSubsetSize: number | null,
  rng: () => number
): { root: TreeNode; featureGains: number[] } {
  const numFeatures = X[0]?.length ?? MODEL_FEATURE_NAMES.length;
  const featureGains = new Array(numFeatures).fill(0);

  const calcWeightedProb = (idxs: number[]) => {
    let posW = 0;
    let totalW = 0;
    for (const idx of idxs) {
      const w = sampleWeights[idx];
      totalW += w;
      if (y[idx] === 1) posW += w;
    }
    return totalW > 0 ? posW / totalW : 0;
  };

  const gini = (posW: number, totalW: number) => {
    if (totalW <= 0) return 0;
    const p = posW / totalW;
    return 1 - p * p - (1 - p) * (1 - p);
  };

  const splitNode = (idxs: number[], depth: number): TreeNode => {
    const prob = calcWeightedProb(idxs);
    if (depth >= maxDepth || idxs.length < minSamplesLeaf * 2 || prob <= 0.001 || prob >= 0.999) {
      return { isLeaf: true, prob };
    }

    let totalW = 0;
    let totalPosW = 0;
    for (const idx of idxs) {
      totalW += sampleWeights[idx];
      if (y[idx] === 1) totalPosW += sampleWeights[idx];
    }
    const parentGini = gini(totalPosW, totalW);

    let candidateFeatures = Array.from({ length: numFeatures }, (_, i) => i);
    if (featureSubsetSize && featureSubsetSize < numFeatures) {
      for (let i = candidateFeatures.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [candidateFeatures[i], candidateFeatures[j]] = [candidateFeatures[j], candidateFeatures[i]];
      }
      candidateFeatures = candidateFeatures.slice(0, featureSubsetSize);
    }

    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    for (const f of candidateFeatures) {
      // Sample up to 24 quantile thresholds for finer split resolution
      const vals = idxs.map((idx) => X[idx][f]).sort((a, b) => a - b);
      const uniqueThresholds: number[] = [];
      const stride = Math.max(1, Math.floor(vals.length / 24));
      for (let k = stride; k < vals.length; k += stride) {
        if (vals[k] !== vals[k - 1]) {
          uniqueThresholds.push((vals[k] + vals[k - 1]) / 2);
        }
      }

      for (const thr of uniqueThresholds) {
        let leftW = 0;
        let leftPosW = 0;
        let leftCount = 0;
        for (const idx of idxs) {
          if (X[idx][f] <= thr) {
            leftW += sampleWeights[idx];
            if (y[idx] === 1) leftPosW += sampleWeights[idx];
            leftCount++;
          }
        }
        const rightCount = idxs.length - leftCount;
        if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;

        const rightW = totalW - leftW;
        const rightPosW = totalPosW - leftPosW;
        const weightedGini =
          (leftW / totalW) * gini(leftPosW, leftW) +
          (rightW / totalW) * gini(rightPosW, rightW);
        const gain = parentGini - weightedGini;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = thr;
        }
      }
    }

    if (bestFeature === -1 || bestGain <= 1e-4) {
      return { isLeaf: true, prob };
    }

    featureGains[bestFeature] += bestGain * totalW;

    const leftIdxs: number[] = [];
    const rightIdxs: number[] = [];
    for (const idx of idxs) {
      if (X[idx][bestFeature] <= bestThreshold) leftIdxs.push(idx);
      else rightIdxs.push(idx);
    }

    return {
      isLeaf: false,
      prob,
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: splitNode(leftIdxs, depth + 1),
      right: splitNode(rightIdxs, depth + 1),
    };
  };

  const root = splitNode(indices, 0);
  return { root, featureGains };
}

function predictTree(node: TreeNode, x: number[]): number {
  let cur = node;
  while (!cur.isLeaf && cur.featureIndex !== undefined && cur.threshold !== undefined) {
    cur = x[cur.featureIndex] <= cur.threshold ? cur.left! : cur.right!;
  }
  return cur.prob;
}

function createSeededRand(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function trainDecisionTreeModel(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[]
): TrainedModelSummary {
  const t0 = performance.now();
  const n = XTrain.length;
  const posCount = Math.max(1, yTrain.filter((y) => y === 1).length);
  const negCount = Math.max(1, n - posCount);
  const posW = n / (2 * posCount);
  const negW = n / (2 * negCount);
  const sampleWeights = yTrain.map((y) => (y === 1 ? posW : negW));
  const indices = Array.from({ length: n }, (_, i) => i);

  const { root, featureGains } = buildDecisionTree(
    XTrain,
    yTrain,
    indices,
    sampleWeights,
    6,
    8,
    null,
    createSeededRand(42)
  );

  const predictProb = (raw: number[]) => predictTree(root, raw);
  const yProbTest = XTest.map((row) => predictProb(row));
  const evalRes = evaluatePredictions(yTest, yProbTest);

  return {
    key: 'decision_tree',
    name: 'Decision Tree',
    shortName: 'DTree',
    ...evalRes,
    trainTimeMs: Number((performance.now() - t0).toFixed(1)),
    featureImportances: normalizeImportances(featureGains),
    predictProb,
  };
}

// 3. Random Forest (Bootstrap Ensemble of 25 Trees with sqrt(d) feature subsampling)
function trainRandomForestModel(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[]
): TrainedModelSummary {
  const t0 = performance.now();
  const n = XTrain.length;
  const d = XTrain[0]?.length ?? MODEL_FEATURE_NAMES.length;
  const posCount = Math.max(1, yTrain.filter((y) => y === 1).length);
  const negCount = Math.max(1, n - posCount);
  const posW = n / (2 * posCount);
  const negW = n / (2 * negCount);
  const sampleWeights = yTrain.map((y) => (y === 1 ? posW : negW));

  const nTrees = 25;
  const subFeatures = Math.max(4, Math.round(Math.sqrt(d)));
  const rng = createSeededRand(101);
  const trees: TreeNode[] = [];
  const totalGains = new Array(d).fill(0);

  for (let t = 0; t < nTrees; t++) {
    const bootIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      bootIndices.push(Math.floor(rng() * n));
    }
    const { root, featureGains } = buildDecisionTree(
      XTrain,
      yTrain,
      bootIndices,
      sampleWeights,
      8,
      5,
      subFeatures,
      rng
    );
    trees.push(root);
    for (let j = 0; j < d; j++) totalGains[j] += featureGains[j];
  }

  const predictProb = (raw: number[]) => {
    let sum = 0;
    for (const tree of trees) sum += predictTree(tree, raw);
    return sum / trees.length;
  };

  const yProbTest = XTest.map((row) => predictProb(row));
  const evalRes = evaluatePredictions(yTest, yProbTest);

  return {
    key: 'random_forest',
    name: 'Random Forest',
    shortName: 'RF',
    ...evalRes,
    trainTimeMs: Number((performance.now() - t0).toFixed(1)),
    featureImportances: normalizeImportances(totalGains),
    predictProb,
  };
}

// 4. Support Vector Machine (Soft-margin Hinge Loss + Balanced Platt Sigmoid Calibration)
function trainSvmModel(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[]
): TrainedModelSummary {
  const t0 = performance.now();
  const n = XTrain.length;
  const d = XTrain[0]?.length ?? MODEL_FEATURE_NAMES.length;
  const weights = new Array(d).fill(0);
  let bias = 0;

  const posCount = Math.max(1, yTrain.filter((y) => y === 1).length);
  const negCount = Math.max(1, n - posCount);
  const posWeight = n / (2 * posCount);
  const negWeight = n / (2 * negCount);

  const lr = 0.05;
  const cPenalty = 1.5;
  const epochs = 260;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW = weights.map((w) => w);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      const ySigned = yTrain[i] === 1 ? 1 : -1;
      const sampleW = yTrain[i] === 1 ? posWeight : negWeight;
      let margin = bias;
      const row = XTrain[i];
      for (let j = 0; j < d; j++) margin += weights[j] * row[j];

      if (ySigned * margin < 1) {
        gradB -= cPenalty * sampleW * ySigned;
        for (let j = 0; j < d; j++) {
          gradW[j] -= (cPenalty * sampleW * ySigned * row[j]) / n;
        }
      }
    }

    const stepLr = lr / (1 + 0.008 * epoch);
    bias -= (stepLr * gradB) / n;
    for (let j = 0; j < d; j++) {
      weights[j] -= stepLr * gradW[j];
    }
  }

  // Platt-like calibration with positive slope alignment & balanced class weights
  const decisionScore = (scaled: number[]) => {
    let z = bias;
    for (let j = 0; j < d; j++) z += weights[j] * scaled[j];
    return z;
  };

  let plattA = 1.5;
  let plattB = 0;
  const trainMargins = XTrain.map(decisionScore);
  for (let iter = 0; iter < 150; iter++) {
    let gA = 0;
    let gB = 0;
    for (let i = 0; i < n; i++) {
      const sampleW = yTrain[i] === 1 ? posWeight : negWeight;
      const p = sigmoid(plattA * trainMargins[i] + plattB);
      const err = (p - yTrain[i]) * sampleW;
      gA += err * trainMargins[i];
      gB += err;
    }
    plattA = Math.max(0.1, plattA - 0.15 * (gA / n));
    plattB -= 0.15 * (gB / n);
  }

  const predictProb = (_raw: number[], scaled: number[]) => {
    const m = decisionScore(scaled);
    return sigmoid(plattA * m + plattB);
  };

  const yProbTest = XTest.map((row) => predictProb(row, row));
  const evalRes = evaluatePredictions(yTest, yProbTest);

  return {
    key: 'svm',
    name: 'Support Vector Machine',
    shortName: 'SVM',
    ...evalRes,
    trainTimeMs: Number((performance.now() - t0).toFixed(1)),
    featureImportances: normalizeImportances(weights),
    predictProb,
  };
}

export function trainAllMlModels(prep: PreprocessingResult): Record<ModelKey, TrainedModelSummary> {
  const lr = trainLogisticRegression(prep.XTrainScaled, prep.yTrain, prep.XTestScaled, prep.yTest);
  const dt = trainDecisionTreeModel(prep.XTrainRaw, prep.yTrain, prep.XTestRaw, prep.yTest);
  const rf = trainRandomForestModel(prep.XTrainRaw, prep.yTrain, prep.XTestRaw, prep.yTest);
  const svm = trainSvmModel(prep.XTrainScaled, prep.yTrain, prep.XTestScaled, prep.yTest);

  return {
    logistic_regression: lr,
    decision_tree: dt,
    random_forest: rf,
    svm,
  };
}

export function predictSinglePatient(
  model: TrainedModelSummary,
  input: PatientPredictionInput,
  prep: PreprocessingResult
): PatientPredictionOutput {
  const rawVec = [
    input.Age,
    input.Gender === 'Female' ? 1 : 0,
    input.Gallstones,
    input.Wall_Thickness_mm,
    input.Polyp_Size_mm,
    input.Chronic_Cholecystitis,
    input.Porcelain_Gallbladder,
    input.Family_History,
    input.BMI,
    input.Bilirubin_mg_dL,
    input.Alkaline_Phosphatase_UL,
    input.Jaundice,
    input.Abdominal_Pain,
  ];

  const scaledVec = rawVec.map(
    (v, j) => (v - prep.scalerMeans[j]) / (prep.scalerStds[j] || 1)
  );

  const prob = model.predictProb(rawVec, scaledVec);
  const probabilityPct = Number((prob * 100).toFixed(1));

  let riskTier: PatientPredictionOutput['riskTier'] = 'Low Estimated Risk';
  if (probabilityPct >= 65) riskTier = 'Elevated Estimated Risk';
  else if (probabilityPct >= 30) riskTier = 'Moderate Estimated Risk';

  // Compute local patient-level contributing features using importance * standardized deviation
  const importanceMap = new Map(
    model.featureImportances.map((item) => [item.feature, item.importancePct])
  );

  const formatVal = (idx: number, val: number): string => {
    if (idx === 1) return val === 1 ? 'Female' : 'Male';
    if ([2, 5, 6, 7, 11, 12].includes(idx)) return val === 1 ? 'Yes' : 'No';
    if (idx === 3 || idx === 4) return `${val.toFixed(1)} mm`;
    if (idx === 9) return `${val.toFixed(2)} mg/dL`;
    if (idx === 10) return `${Math.round(val)} U/L`;
    return String(Number(val.toFixed(1)));
  };

  const contributions: FeatureContribution[] = MODEL_FEATURE_NAMES.map((name, idx) => {
    const imp = importanceMap.get(name) ?? 0;
    const zScore = scaledVec[idx];
    const impact = Number((zScore * (imp / 100) * 100).toFixed(1));
    let direction: FeatureContribution['direction'] = 'neutral';
    if (impact > 0.6) direction = 'increases';
    else if (impact < -0.6) direction = 'decreases';

    return {
      feature: name,
      patientValue: formatVal(idx, rawVec[idx]),
      cohortMean: formatVal(idx, prep.scalerMeans[idx]),
      impactScore: impact,
      direction,
    };
  })
    .sort((a, b) => Math.abs(b.impactScore) - Math.abs(a.impactScore))
    .slice(0, 6);

  return {
    modelName: model.name,
    probabilityPct,
    riskTier,
    contributions,
  };
}
