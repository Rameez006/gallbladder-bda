export type GenderType = 'Female' | 'Male';

export interface RawPatientRecord {
  Patient_ID: string;
  Age: number | null;
  Gender: GenderType | null;
  Gallstones: number | null;
  Wall_Thickness_mm: number | null;
  Polyp_Size_mm: number | null;
  Chronic_Cholecystitis: number | null;
  Porcelain_Gallbladder: number | null;
  Family_History: number | null;
  BMI: number | null;
  Bilirubin_mg_dL: number | null;
  Alkaline_Phosphatase_UL: number | null;
  Jaundice: number | null;
  Abdominal_Pain: number | null;
  Cancer_Risk: number | null;
}

export interface CleanedPatientRecord {
  Patient_ID: string;
  Age: number;
  Gender: GenderType;
  Gallstones: number;
  Wall_Thickness_mm: number;
  Polyp_Size_mm: number;
  Chronic_Cholecystitis: number;
  Porcelain_Gallbladder: number;
  Family_History: number;
  BMI: number;
  Bilirubin_mg_dL: number;
  Alkaline_Phosphatase_UL: number;
  Jaundice: number;
  Abdominal_Pain: number;
  Cancer_Risk: number;
}

export interface ColumnMissingStat {
  column: string;
  missingCount: number;
  missingPct: number;
  imputeStrategy: string;
  imputedValue: string;
}

export interface PreprocessingResult {
  cleanedRecords: CleanedPatientRecord[];
  rawRows: number;
  duplicatesRemoved: number;
  missingTargetDropped: number;
  totalMissingCells: number;
  cleanedRows: number;
  trainRows: number;
  testRows: number;
  positiveRatePct: number;
  columnMissingStats: ColumnMissingStat[];
  featureNames: string[];
  XTrainRaw: number[][];
  XTestRaw: number[][];
  XTrainScaled: number[][];
  XTestScaled: number[][];
  yTrain: number[];
  yTest: number[];
  scalerMeans: number[];
  scalerStds: number[];
  imputationMedians: Record<string, number>;
  genderMode: GenderType;
}

export interface SparkPartitionStat {
  partitionId: number;
  rowCount: number;
  positiveCount: number;
  gallstoneCount: number;
  avgWallThickness: number;
  highRiskCohortCount: number;
  mapDurationMs: number;
}

export interface SparkGroupSummary {
  groupLabel: string;
  recordCount: number;
  avgAge: number;
  avgWallThickness: number;
  elevatedRiskPct: number;
}

export interface SparkAnalyticsResult {
  totalRecords: number;
  partitionsUsed: number;
  highRiskFilterCount: number;
  totalDurationMs: number;
  partitionStats: SparkPartitionStat[];
  gallstoneAggregation: SparkGroupSummary[];
  ageBracketAggregation: SparkGroupSummary[];
  genderAggregation: SparkGroupSummary[];
}

export interface AgeBinStat {
  label: string;
  lowRiskCount: number;
  elevatedRiskCount: number;
}

export interface RiskFactorPrevalence {
  factor: string;
  withFactorPct: number;
  withoutFactorPct: number;
  withCount: number;
  withoutCount: number;
}

export interface BoxPlotStats {
  label: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
}

export interface EdaAnalyticsResult {
  ageDistribution: AgeBinStat[];
  ageDistribution5Yr: AgeBinStat[];
  riskFactorComparison: RiskFactorPrevalence[];
  wallThicknessByRisk: BoxPlotStats[];
  polypSizeByRisk: BoxPlotStats[];
  correlationLabels: string[];
  correlationMatrix: number[][];
}

export type ModelKey = 'logistic_regression' | 'decision_tree' | 'random_forest' | 'svm';

export interface FeatureImportanceItem {
  feature: string;
  importancePct: number;
}

export interface ConfusionMatrixCounts {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface TrainedModelSummary {
  key: ModelKey;
  name: string;
  shortName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  trainTimeMs: number;
  confusionMatrix: ConfusionMatrixCounts;
  rocCurve: RocPoint[];
  featureImportances: FeatureImportanceItem[];
  predictProb: (rawFeatures: number[], scaledFeatures: number[]) => number;
}

export interface PatientPredictionInput {
  Age: number;
  Gender: GenderType;
  Gallstones: number;
  Wall_Thickness_mm: number;
  Polyp_Size_mm: number;
  Chronic_Cholecystitis: number;
  Porcelain_Gallbladder: number;
  Family_History: number;
  BMI: number;
  Bilirubin_mg_dL: number;
  Alkaline_Phosphatase_UL: number;
  Jaundice: number;
  Abdominal_Pain: number;
}

export interface FeatureContribution {
  feature: string;
  patientValue: string;
  cohortMean: string;
  impactScore: number;
  direction: 'increases' | 'decreases' | 'neutral';
}

export interface PatientPredictionOutput {
  modelName: string;
  probabilityPct: number;
  riskTier: 'Low Estimated Risk' | 'Moderate Estimated Risk' | 'Elevated Estimated Risk';
  contributions: FeatureContribution[];
}
