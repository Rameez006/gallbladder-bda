import {
  CleanedPatientRecord,
  SparkAnalyticsResult,
  SparkGroupSummary,
  SparkPartitionStat,
} from '../types/medical';

/**
 * Executes a genuine MapReduce-style partitioned aggregation over the dataset
 * to demonstrate how Apache Spark (PySpark) distributes workloads across worker partitions.
 */
export function runSparkPartitionedAnalytics(
  records: CleanedPatientRecord[],
  partitions: number
): SparkAnalyticsResult {
  const startTotal = performance.now();
  const totalRecords = records.length;
  const chunkSize = Math.ceil(totalRecords / partitions);

  const partitionStats: SparkPartitionStat[] = [];

  // Map Phase: process each partition slice independently
  interface PartialAgg {
    count: number;
    ageSum: number;
    wallSum: number;
    posCount: number;
  }

  const gallstoneMap = new Map<string, PartialAgg>();
  const ageBracketMap = new Map<string, PartialAgg>();
  const genderMap = new Map<string, PartialAgg>();

  const accumulate = (map: Map<string, PartialAgg>, key: string, r: CleanedPatientRecord) => {
    const cur = map.get(key) ?? { count: 0, ageSum: 0, wallSum: 0, posCount: 0 };
    cur.count += 1;
    cur.ageSum += r.Age;
    cur.wallSum += r.Wall_Thickness_mm;
    cur.posCount += r.Cancer_Risk;
    map.set(key, cur);
  };

  let highRiskFilterCount = 0;

  for (let p = 0; p < partitions; p++) {
    const mapStart = performance.now();
    const slice = records.slice(p * chunkSize, (p + 1) * chunkSize);

    let posCount = 0;
    let gallstoneCount = 0;
    let wallSum = 0;
    let localHighRisk = 0;

    for (const r of slice) {
      posCount += r.Cancer_Risk;
      gallstoneCount += r.Gallstones;
      wallSum += r.Wall_Thickness_mm;

      if (r.Age >= 50 && r.Gallstones === 1 && r.Wall_Thickness_mm >= 3.5) {
        localHighRisk += 1;
      }

      accumulate(gallstoneMap, r.Gallstones === 1 ? 'Gallstones Present (1)' : 'No Gallstones (0)', r);

      const bracket =
        r.Age < 40 ? 'Under 40' : r.Age < 55 ? '40 - 54' : r.Age < 70 ? '55 - 69' : '70+';
      accumulate(ageBracketMap, bracket, r);

      accumulate(genderMap, r.Gender, r);
    }

    highRiskFilterCount += localHighRisk;
    const mapEnd = performance.now();

    partitionStats.push({
      partitionId: p,
      rowCount: slice.length,
      positiveCount: posCount,
      gallstoneCount,
      avgWallThickness: slice.length > 0 ? Number((wallSum / slice.length).toFixed(2)) : 0,
      highRiskCohortCount: localHighRisk,
      mapDurationMs: Number((mapEnd - mapStart).toFixed(2)),
    });
  }

  // Reduce Phase: merge partition maps into final Spark DataFrame summaries
  const finalize = (map: Map<string, PartialAgg>, order?: string[]): SparkGroupSummary[] => {
    const entries = Array.from(map.entries()).map(([groupLabel, agg]) => ({
      groupLabel,
      recordCount: agg.count,
      avgAge: Number((agg.ageSum / Math.max(1, agg.count)).toFixed(1)),
      avgWallThickness: Number((agg.wallSum / Math.max(1, agg.count)).toFixed(2)),
      elevatedRiskPct: Number(((agg.posCount / Math.max(1, agg.count)) * 100).toFixed(2)),
    }));

    if (order) {
      entries.sort((a, b) => order.indexOf(a.groupLabel) - order.indexOf(b.groupLabel));
    }
    return entries;
  };

  const totalDurationMs = Number((performance.now() - startTotal).toFixed(2));

  return {
    totalRecords,
    partitionsUsed: partitions,
    highRiskFilterCount,
    totalDurationMs,
    partitionStats,
    gallstoneAggregation: finalize(gallstoneMap, ['No Gallstones (0)', 'Gallstones Present (1)']),
    ageBracketAggregation: finalize(ageBracketMap, ['Under 40', '40 - 54', '55 - 69', '70+']),
    genderAggregation: finalize(genderMap, ['Female', 'Male']),
  };
}
