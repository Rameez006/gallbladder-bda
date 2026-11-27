import {
  CleanedPatientRecord,
  ColumnMissingStat,
  GenderType,
  PreprocessingResult,
  RawPatientRecord,
} from '../types/medical';

export const MODEL_FEATURE_NAMES = [
  'Age',
  'Gender (Female=1)',
  'Gallstones',
  'Wall Thickness (mm)',
  'Polyp Size (mm)',
  'Chronic Cholecystitis',
  'Porcelain Gallbladder',
  'Family History',
  'BMI',
  'Bilirubin (mg/dL)',
  'Alkaline Phosphatase (U/L)',
  'Jaundice',
  'Abdominal Pain',
];

function computeMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const copy = [...arr];
  let s = seed >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function extractFeatureVector(r: CleanedPatientRecord): number[] {
  return [
    r.Age,
    r.Gender === 'Female' ? 1 : 0,
    r.Gallstones,
    r.Wall_Thickness_mm,
    r.Polyp_Size_mm,
    r.Chronic_Cholecystitis,
    r.Porcelain_Gallbladder,
    r.Family_History,
    r.BMI,
    r.Bilirubin_mg_dL,
    r.Alkaline_Phosphatase_UL,
    r.Jaundice,
    r.Abdominal_Pain,
  ];
}

export function cleanAndPreprocess(
  rawRecords: RawPatientRecord[],
  testRatio = 0.2,
  seed = 42
): PreprocessingResult {
  const rawRows = rawRecords.length;

  // Count missing cells in raw data
  const inspectColumns: Array<{ key: keyof RawPatientRecord; label: string; isCategorical?: boolean }> = [
    { key: 'Age', label: 'Age' },
    { key: 'Gender', label: 'Gender', isCategorical: true },
    { key: 'Gallstones', label: 'Gallstones' },
    { key: 'Wall_Thickness_mm', label: 'Wall_Thickness_mm' },
    { key: 'Polyp_Size_mm', label: 'Polyp_Size_mm' },
    { key: 'BMI', label: 'BMI' },
    { key: 'Bilirubin_mg_dL', label: 'Bilirubin_mg_dL' },
    { key: 'Alkaline_Phosphatase_UL', label: 'Alkaline_Phosphatase_UL' },
  ];

  let totalMissingCells = 0;
  for (const r of rawRecords) {
    for (const col of Object.keys(r) as Array<keyof RawPatientRecord>) {
      if (r[col] === null || r[col] === undefined) totalMissingCells++;
    }
  }

  // 1. Remove exact duplicate records
  const seen = new Set<string>();
  const deduped: RawPatientRecord[] = [];
  for (const r of rawRecords) {
    const sig = JSON.stringify([
      r.Patient_ID,
      r.Age,
      r.Gender,
      r.Gallstones,
      r.Wall_Thickness_mm,
      r.Polyp_Size_mm,
      r.BMI,
      r.Cancer_Risk,
    ]);
    if (!seen.has(sig)) {
      seen.add(sig);
      deduped.push(r);
    }
  }
  const duplicatesRemoved = rawRows - deduped.length;

  // 2. Drop rows missing target Cancer_Risk
  const withTarget = deduped.filter((r) => r.Cancer_Risk === 0 || r.Cancer_Risk === 1);
  const missingTargetDropped = deduped.length - withTarget.length;

  // 3. Stratified Train / Test split BEFORE fitting imputation or scaling (zero data leakage)
  const posRows = seededShuffle(
    withTarget.filter((r) => r.Cancer_Risk === 1),
    seed
  );
  const negRows = seededShuffle(
    withTarget.filter((r) => r.Cancer_Risk === 0),
    seed + 1
  );

  const posTestCount = Math.max(1, Math.round(posRows.length * testRatio));
  const negTestCount = Math.max(1, Math.round(negRows.length * testRatio));

  const rawTest = seededShuffle(
    [...posRows.slice(0, posTestCount), ...negRows.slice(0, negTestCount)],
    seed + 2
  );
  const rawTrain = seededShuffle(
    [...posRows.slice(posTestCount), ...negRows.slice(negTestCount)],
    seed + 3
  );

  // 4. Compute imputation medians & mode STRICTLY on rawTrain
  const numericDefaults: Record<string, number> = {
    Age: 52,
    Gallstones: 0,
    Wall_Thickness_mm: 2.8,
    Polyp_Size_mm: 0,
    Chronic_Cholecystitis: 0,
    Porcelain_Gallbladder: 0,
    Family_History: 0,
    BMI: 26.5,
    Bilirubin_mg_dL: 0.9,
    Alkaline_Phosphatase_UL: 95,
    Jaundice: 0,
    Abdominal_Pain: 0,
  };

  const imputationMedians: Record<string, number> = {};
  for (const key of Object.keys(numericDefaults)) {
    const vals = rawTrain
      .map((r) => r[key as keyof RawPatientRecord])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    imputationMedians[key] = vals.length > 0 ? Number(computeMedian(vals).toFixed(2)) : numericDefaults[key];
  }

  let femaleCount = 0;
  let maleCount = 0;
  for (const r of rawTrain) {
    if (r.Gender === 'Female') femaleCount++;
    else if (r.Gender === 'Male') maleCount++;
  }
  const genderMode: GenderType = femaleCount >= maleCount ? 'Female' : 'Male';

  const imputeRecord = (r: RawPatientRecord): CleanedPatientRecord => ({
    Patient_ID: r.Patient_ID,
    Age: r.Age ?? imputationMedians.Age,
    Gender: r.Gender ?? genderMode,
    Gallstones: r.Gallstones ?? imputationMedians.Gallstones,
    Wall_Thickness_mm: r.Wall_Thickness_mm ?? imputationMedians.Wall_Thickness_mm,
    Polyp_Size_mm: r.Polyp_Size_mm ?? imputationMedians.Polyp_Size_mm,
    Chronic_Cholecystitis: r.Chronic_Cholecystitis ?? imputationMedians.Chronic_Cholecystitis,
    Porcelain_Gallbladder: r.Porcelain_Gallbladder ?? imputationMedians.Porcelain_Gallbladder,
    Family_History: r.Family_History ?? imputationMedians.Family_History,
    BMI: r.BMI ?? imputationMedians.BMI,
    Bilirubin_mg_dL: r.Bilirubin_mg_dL ?? imputationMedians.Bilirubin_mg_dL,
    Alkaline_Phosphatase_UL: r.Alkaline_Phosphatase_UL ?? imputationMedians.Alkaline_Phosphatase_UL,
    Jaundice: r.Jaundice ?? imputationMedians.Jaundice,
    Abdominal_Pain: r.Abdominal_Pain ?? imputationMedians.Abdominal_Pain,
    Cancer_Risk: r.Cancer_Risk ?? 0,
  });

  const trainClean = rawTrain.map(imputeRecord);
  const testClean = rawTest.map(imputeRecord);
  const cleanedRecords = withTarget.map(imputeRecord);

  const columnMissingStats: ColumnMissingStat[] = inspectColumns.map((col) => {
    const missingCount = deduped.filter((r) => r[col.key] === null || r[col.key] === undefined).length;
    return {
      column: col.label,
      missingCount,
      missingPct: Number(((missingCount / Math.max(1, deduped.length)) * 100).toFixed(2)),
      imputeStrategy: col.isCategorical ? 'Mode (Train Split)' : 'Median (Train Split)',
      imputedValue: col.isCategorical
        ? genderMode
        : String(imputationMedians[col.key as string] ?? 0),
    };
  });

  // 5. Compute Standardization (Z-score) parameters strictly on XTrainRaw
  const XTrainRaw = trainClean.map(extractFeatureVector);
  const XTestRaw = testClean.map(extractFeatureVector);
  const yTrain = trainClean.map((r) => r.Cancer_Risk);
  const yTest = testClean.map((r) => r.Cancer_Risk);

  const numFeatures = MODEL_FEATURE_NAMES.length;
  const scalerMeans = new Array(numFeatures).fill(0);
  const scalerStds = new Array(numFeatures).fill(1);

  for (let j = 0; j < numFeatures; j++) {
    let sum = 0;
    for (let i = 0; i < XTrainRaw.length; i++) sum += XTrainRaw[i][j];
    const mean = sum / Math.max(1, XTrainRaw.length);
    scalerMeans[j] = mean;

    let sq = 0;
    for (let i = 0; i < XTrainRaw.length; i++) {
      const diff = XTrainRaw[i][j] - mean;
      sq += diff * diff;
    }
    const std = Math.sqrt(sq / Math.max(1, XTrainRaw.length));
    scalerStds[j] = std > 1e-6 ? std : 1;
  }

  const scaleMatrix = (mat: number[][]): number[][] =>
    mat.map((row) => row.map((v, j) => (v - scalerMeans[j]) / scalerStds[j]));

  const XTrainScaled = scaleMatrix(XTrainRaw);
  const XTestScaled = scaleMatrix(XTestRaw);

  const posTotal = cleanedRecords.filter((r) => r.Cancer_Risk === 1).length;
  const positiveRatePct = Number(((posTotal / Math.max(1, cleanedRecords.length)) * 100).toFixed(1));

  return {
    cleanedRecords,
    rawRows,
    duplicatesRemoved,
    missingTargetDropped,
    totalMissingCells,
    cleanedRows: cleanedRecords.length,
    trainRows: trainClean.length,
    testRows: testClean.length,
    positiveRatePct,
    columnMissingStats,
    featureNames: MODEL_FEATURE_NAMES,
    XTrainRaw,
    XTestRaw,
    XTrainScaled,
    XTestScaled,
    yTrain,
    yTest,
    scalerMeans,
    scalerStds,
    imputationMedians,
    genderMode,
  };
}
