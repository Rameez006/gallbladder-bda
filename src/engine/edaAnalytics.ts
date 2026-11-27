import {
  AgeBinStat,
  BoxPlotStats,
  CleanedPatientRecord,
  EdaAnalyticsResult,
  RiskFactorPrevalence,
} from '../types/medical';

function computeBoxStats(label: string, values: number[]): BoxPlotStats {
  if (values.length === 0) {
    return { label, min: 0, q1: 0, median: 0, q3: 0, max: 0, mean: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = (q: number) => {
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    }
    return sorted[base];
  };
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  return {
    label,
    min: Number(sorted[0].toFixed(2)),
    q1: Number(quantile(0.25).toFixed(2)),
    median: Number(quantile(0.5).toFixed(2)),
    q3: Number(quantile(0.75).toFixed(2)),
    max: Number(sorted[sorted.length - 1].toFixed(2)),
    mean: Number((sum / sorted.length).toFixed(2)),
  };
}

function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const denom = Math.sqrt(denX * denY);
  return denom > 1e-9 ? Number((num / denom).toFixed(2)) : 0;
}

export function computeEdaAnalytics(records: CleanedPatientRecord[]): EdaAnalyticsResult {
  // 1. Age distribution across 7 clinical brackets
  const ageBinsDef = [
    { label: '20-29', min: 0, max: 29 },
    { label: '30-39', min: 30, max: 39 },
    { label: '40-49', min: 40, max: 49 },
    { label: '50-59', min: 50, max: 59 },
    { label: '60-69', min: 60, max: 69 },
    { label: '70-79', min: 70, max: 79 },
    { label: '80+', min: 80, max: 200 },
  ];

  const ageDistribution: AgeBinStat[] = ageBinsDef.map((bin) => {
    const subset = records.filter((r) => r.Age >= bin.min && r.Age <= bin.max);
    return {
      label: bin.label,
      lowRiskCount: subset.filter((r) => r.Cancer_Risk === 0).length,
      elevatedRiskCount: subset.filter((r) => r.Cancer_Risk === 1).length,
    };
  });

  // 1b. Finer 5-year age brackets for high-resolution histogram
  const ageBins5YrDef = [
    { label: '20-24', min: 0, max: 24 },
    { label: '25-29', min: 25, max: 29 },
    { label: '30-34', min: 30, max: 34 },
    { label: '35-39', min: 35, max: 39 },
    { label: '40-44', min: 40, max: 44 },
    { label: '45-49', min: 45, max: 49 },
    { label: '50-54', min: 50, max: 54 },
    { label: '55-59', min: 55, max: 59 },
    { label: '60-64', min: 60, max: 64 },
    { label: '65-69', min: 65, max: 69 },
    { label: '70-74', min: 70, max: 74 },
    { label: '75-79', min: 75, max: 79 },
    { label: '80+', min: 80, max: 200 },
  ];

  const ageDistribution5Yr: AgeBinStat[] = ageBins5YrDef.map((bin) => {
    const subset = records.filter((r) => r.Age >= bin.min && r.Age <= bin.max);
    return {
      label: bin.label,
      lowRiskCount: subset.filter((r) => r.Cancer_Risk === 0).length,
      elevatedRiskCount: subset.filter((r) => r.Cancer_Risk === 1).length,
    };
  });

  // 2. Binary risk factor prevalence comparisons
  const calcPrevalence = (
    factor: string,
    predicate: (r: CleanedPatientRecord) => boolean
  ): RiskFactorPrevalence => {
    const withGroup = records.filter(predicate);
    const withoutGroup = records.filter((r) => !predicate(r));
    const withPos = withGroup.filter((r) => r.Cancer_Risk === 1).length;
    const withoutPos = withoutGroup.filter((r) => r.Cancer_Risk === 1).length;
    return {
      factor,
      withFactorPct: withGroup.length > 0 ? Number(((withPos / withGroup.length) * 100).toFixed(1)) : 0,
      withoutFactorPct:
        withoutGroup.length > 0 ? Number(((withoutPos / withoutGroup.length) * 100).toFixed(1)) : 0,
      withCount: withGroup.length,
      withoutCount: withoutGroup.length,
    };
  };

  const riskFactorComparison: RiskFactorPrevalence[] = [
    calcPrevalence('Gallstones', (r) => r.Gallstones === 1),
    calcPrevalence('Chronic Cholecystitis', (r) => r.Chronic_Cholecystitis === 1),
    calcPrevalence('Porcelain Gallbladder', (r) => r.Porcelain_Gallbladder === 1),
    calcPrevalence('Family History', (r) => r.Family_History === 1),
    calcPrevalence('Female Sex', (r) => r.Gender === 'Female'),
    calcPrevalence('Jaundice', (r) => r.Jaundice === 1),
  ];

  // 3. Wall thickness & Polyp size box statistics
  const lowRecords = records.filter((r) => r.Cancer_Risk === 0);
  const highRecords = records.filter((r) => r.Cancer_Risk === 1);

  const wallThicknessByRisk: BoxPlotStats[] = [
    computeBoxStats(
      'Low Risk (0)',
      lowRecords.map((r) => r.Wall_Thickness_mm)
    ),
    computeBoxStats(
      'Elevated Risk (1)',
      highRecords.map((r) => r.Wall_Thickness_mm)
    ),
  ];

  const polypSizeByRisk: BoxPlotStats[] = [
    computeBoxStats(
      'Low Risk (0)',
      lowRecords.map((r) => r.Polyp_Size_mm)
    ),
    computeBoxStats(
      'Elevated Risk (1)',
      highRecords.map((r) => r.Polyp_Size_mm)
    ),
  ];

  // 4. Pearson Correlation Matrix across 9 key clinical features + Target
  const corrSpecs: Array<{ label: string; getter: (r: CleanedPatientRecord) => number }> = [
    { label: 'Age', getter: (r) => r.Age },
    { label: 'Gallstones', getter: (r) => r.Gallstones },
    { label: 'Wall Thick', getter: (r) => r.Wall_Thickness_mm },
    { label: 'Polyp Size', getter: (r) => r.Polyp_Size_mm },
    { label: 'Cholecystitis', getter: (r) => r.Chronic_Cholecystitis },
    { label: 'Porcelain GB', getter: (r) => r.Porcelain_Gallbladder },
    { label: 'Family Hist', getter: (r) => r.Family_History },
    { label: 'Bilirubin', getter: (r) => r.Bilirubin_mg_dL },
    { label: 'Alk Phos', getter: (r) => r.Alkaline_Phosphatase_UL },
    { label: 'Cancer Risk', getter: (r) => r.Cancer_Risk },
  ];

  const vectors = corrSpecs.map((spec) => records.map(spec.getter));
  const correlationMatrix: number[][] = vectors.map((vecA, i) =>
    vectors.map((vecB, j) => (i === j ? 1 : pearsonCorrelation(vecA, vecB)))
  );

  return {
    ageDistribution,
    ageDistribution5Yr,
    riskFactorComparison,
    wallThicknessByRisk,
    polypSizeByRisk,
    correlationLabels: corrSpecs.map((s) => s.label),
    correlationMatrix,
  };
}
