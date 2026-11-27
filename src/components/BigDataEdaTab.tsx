import React, { useState } from 'react';
import { Play } from 'lucide-react';
import { EdaAnalyticsResult, SparkAnalyticsResult } from '../types/medical';

interface BigDataEdaTabProps {
  sparkPartitions: number;
  onChangeSparkPartitions: (p: number) => void;
  onRunSparkJob: () => void;
  sparkResult: SparkAnalyticsResult;
  eda: EdaAnalyticsResult;
}

export const BigDataEdaTab: React.FC<BigDataEdaTabProps> = ({
  sparkPartitions,
  onChangeSparkPartitions,
  onRunSparkJob,
  sparkResult,
  eda,
}) => {
  const [ageResolution, setAgeResolution] = useState<'5yr' | '10yr'>('5yr');
  const [hoveredAgeBin, setHoveredAgeBin] = useState<number | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  const activeAgeBins = ageResolution === '5yr' ? eda.ageDistribution5Yr : eda.ageDistribution;

  const rawMaxAge = Math.max(
    1,
    ...activeAgeBins.map((b) => Math.max(b.lowRiskCount, b.elevatedRiskCount))
  );
  // Round Y-axis ceiling cleanly to next nice multiple of 25 or 50 for accurate grid ticks
  const stepUnit = rawMaxAge > 200 ? 50 : 25;
  const maxAgeCount = Math.ceil(rawMaxAge / stepUnit) * stepUnit;

  const maxPartitionRows = Math.max(1, ...sparkResult.partitionStats.map((p) => p.rowCount));

  const corrColor = (val: number): string => {
    if (val >= 0.55) return '#1E3A5F';
    if (val >= 0.3) return '#3B5E8C';
    if (val >= 0.12) return '#CBD5E1';
    if (val <= -0.3) return '#881337';
    if (val <= -0.12) return '#FECDD3';
    return '#F3EFE7';
  };

  const corrTextColor = (val: number): string => {
    if (val >= 0.3 || val <= -0.3) return '#FFFFFF';
    return '#1C1917';
  };

  const chartLeft = 48;
  const chartWidth = 500;
  const groupSlot = chartWidth / activeAgeBins.length;
  const barWidth = ageResolution === '5yr' ? Math.max(8, groupSlot * 0.34) : 22;

  return (
    <div className="space-y-8">
      {/* Module 3: PySpark Distributed Analytics */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <div className="flex flex-col gap-4 border-b border-[#ECE7DE] pb-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-semibold tracking-wide text-[#1E3A5F]">
              Section II · Apache Spark (PySpark) Distributed Analytics
            </p>
            <h2 className="mt-1 font-display text-2xl font-bold text-stone-900">
              Partitioned MapReduce Aggregation &amp; Cohort Filtering
            </h2>
            <p className="mt-1 text-xs text-stone-600">
              Splits {sparkResult.totalRecords.toLocaleString()} cleaned medical records across{' '}
              {sparkPartitions} worker partitions, executes local Map aggregations, and reduces
              global clinical groupBy tables.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <div className="flex items-center gap-1 border border-[#DFD9CE] bg-[#F3EFE7] p-1">
              {[2, 4, 8, 16].map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => onChangeSparkPartitions(p)}
                  className={`px-3 py-1.5 text-xs font-mono tabular-nums font-medium transition-colors whitespace-nowrap shrink-0 ${
                    sparkPartitions === p
                      ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                >
                  {p} partitions
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={onRunSparkJob}
              className="inline-flex items-center gap-2 rounded-sm bg-[#1E3A5F] px-4 py-2 text-xs font-semibold text-white hover:bg-[#162C46] whitespace-nowrap shrink-0"
            >
              <Play className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Run Spark Aggregation</span>
            </button>
          </div>
        </div>

        {/* Spark Summary KPIs */}
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-4">
          <div className="border border-[#DFD9CE] bg-[#FAF8F4] p-4">
            <p className="text-xs text-stone-500">Distributed Records</p>
            <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-stone-900">
              {sparkResult.totalRecords.toLocaleString()}
            </p>
          </div>
          <div className="border border-[#DFD9CE] bg-[#FAF8F4] p-4">
            <p className="text-xs text-stone-500">Active Spark Partitions</p>
            <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-stone-900">
              {sparkResult.partitionsUsed}
            </p>
          </div>
          <div className="border border-[#DFD9CE] bg-[#FAF8F4] p-4">
            <p className="text-xs text-stone-500">High-Risk Filter Match</p>
            <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-[#881337]">
              {sparkResult.highRiskFilterCount.toLocaleString()}
            </p>
            <p className="mt-0.5 text-[11px] text-stone-500">
              Age ≥ 50 · Stones=1 · Wall ≥ 3.5mm
            </p>
          </div>
          <div className="border border-[#DFD9CE] bg-[#FAF8F4] p-4">
            <p className="text-xs text-stone-500">MapReduce Elapsed Time</p>
            <p className="mt-1 font-mono text-xl font-semibold tabular-nums text-[#1E3A5F]">
              {sparkResult.totalDurationMs.toFixed(2)} ms
            </p>
          </div>
        </div>

        {/* Partition Workload Table & Spark Aggregations */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <h3 className="font-display text-sm font-semibold text-stone-900">
              Fig. 1 — Partition Map Phase Distribution
            </h3>
            <div className="mt-3 space-y-2.5">
              {sparkResult.partitionStats.map((part) => {
                const widthPct = Math.round((part.rowCount / maxPartitionRows) * 100);
                return (
                  <div key={part.partitionId} className="space-y-1 text-xs">
                    <div className="flex items-center justify-between font-mono tabular-nums">
                      <span className="text-stone-800">Partition #{part.partitionId}</span>
                      <span className="text-stone-500">
                        {part.rowCount} rows · {part.positiveCount} elevated ·{' '}
                        {part.avgWallThickness} mm
                      </span>
                    </div>
                    <div className="h-2 w-full bg-[#ECE7DE]">
                      <div
                        className="h-2 bg-[#1E3A5F] transition-all"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="space-y-5 lg:col-span-7">
            <div>
              <h3 className="font-display text-sm font-semibold text-stone-900">
                Table 4. Spark groupBy(&quot;Gallstones&quot;) Aggregation
              </h3>
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                      <th className="py-2 pl-2 pr-3 font-semibold">Group</th>
                      <th className="py-2 px-2 text-right font-semibold">Records</th>
                      <th className="py-2 px-2 text-right font-semibold">Avg Age</th>
                      <th className="py-2 px-2 text-right font-semibold">Avg Wall (mm)</th>
                      <th className="py-2 pl-2 pr-2 text-right font-semibold">Elevated Risk %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#ECE7DE] font-mono tabular-nums">
                    {sparkResult.gallstoneAggregation.map((row) => (
                      <tr key={row.groupLabel} className="hover:bg-[#FAF8F4]">
                        <td className="py-2 pl-2 pr-3 font-sans font-medium text-stone-800">
                          {row.groupLabel}
                        </td>
                        <td className="py-2 px-2 text-right">{row.recordCount.toLocaleString()}</td>
                        <td className="py-2 px-2 text-right">{row.avgAge}</td>
                        <td className="py-2 px-2 text-right">{row.avgWallThickness}</td>
                        <td className="py-2 pl-2 pr-2 text-right font-semibold text-[#1E3A5F]">
                          {row.elevatedRiskPct}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="font-display text-sm font-semibold text-stone-900">
                Table 5. Spark groupBy(&quot;Age_Bracket&quot;) Aggregation
              </h3>
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                      <th className="py-2 pl-2 pr-3 font-semibold">Age Bracket</th>
                      <th className="py-2 px-2 text-right font-semibold">Records</th>
                      <th className="py-2 px-2 text-right font-semibold">Avg Wall (mm)</th>
                      <th className="py-2 pl-2 pr-2 text-right font-semibold">Elevated Risk %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#ECE7DE] font-mono tabular-nums">
                    {sparkResult.ageBracketAggregation.map((row) => (
                      <tr key={row.groupLabel} className="hover:bg-[#FAF8F4]">
                        <td className="py-2 pl-2 pr-3 font-sans font-medium text-stone-800">
                          {row.groupLabel}
                        </td>
                        <td className="py-2 px-2 text-right">{row.recordCount.toLocaleString()}</td>
                        <td className="py-2 px-2 text-right">{row.avgWallThickness}</td>
                        <td className="py-2 pl-2 pr-2 text-right font-semibold text-[#1E3A5F]">
                          {row.elevatedRiskPct}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Module 4: Exploratory Data Analysis */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Chart 1: High-Resolution Age Distribution Grouped Histogram */}
        <div className="border border-[#DFD9CE] bg-white p-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-[#1E3A5F]">
                Section III · Exploratory Data Analysis
              </p>
              <h3 className="mt-0.5 font-display text-lg font-semibold text-stone-900">
                Fig. 2 — Age Distribution by Risk Class
              </h3>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              <div className="flex items-center gap-1 border border-[#DFD9CE] bg-[#F3EFE7] p-0.5">
                <button
                  type="button"
                  onClick={() => {
                    setAgeResolution('5yr');
                    setHoveredAgeBin(null);
                  }}
                  className={`px-2 py-1 text-[11px] font-mono font-medium transition-colors ${
                    ageResolution === '5yr'
                      ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                      : 'text-stone-600'
                  }`}
                >
                  5-yr bins
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAgeResolution('10yr');
                    setHoveredAgeBin(null);
                  }}
                  className={`px-2 py-1 text-[11px] font-mono font-medium transition-colors ${
                    ageResolution === '10yr'
                      ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                      : 'text-stone-600'
                  }`}
                >
                  10-yr bins
                </button>
              </div>
            </div>
          </div>

          <div className="mt-2 flex items-center gap-4 text-xs text-stone-600">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 bg-[#1E3A5F]" /> Low Risk (0)
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 bg-[#881337]" /> Elevated Risk (1)
            </span>
          </div>

          <div className="mt-3">
            <svg
              viewBox="0 0 568 258"
              className="h-auto w-full"
              role="img"
              aria-label="Age distribution grouped bar chart"
            >
              {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
                const y = 200 - tick * 170;
                const val = Math.round(tick * maxAgeCount);
                return (
                  <g key={tick}>
                    <line
                      x1={chartLeft}
                      y1={y}
                      x2={chartLeft + chartWidth}
                      y2={y}
                      stroke="#ECE7DE"
                      strokeWidth={1}
                    />
                    <text
                      x={chartLeft - 6}
                      y={y + 3}
                      textAnchor="end"
                      className="fill-stone-500 text-[10px] font-mono"
                    >
                      {val}
                    </text>
                  </g>
                );
              })}

              {/* Y-axis baseline */}
              <line x1={chartLeft} y1={30} x2={chartLeft} y2={200} stroke="#D6D3CD" strokeWidth={1} />
              <line
                x1={chartLeft}
                y1={200}
                x2={chartLeft + chartWidth}
                y2={200}
                stroke="#D6D3CD"
                strokeWidth={1}
              />

              {activeAgeBins.map((bin, idx) => {
                const slotCenter = chartLeft + idx * groupSlot + groupSlot / 2;
                const lowH = (bin.lowRiskCount / maxAgeCount) * 170;
                const highH = (bin.elevatedRiskCount / maxAgeCount) * 170;
                const isHovered = hoveredAgeBin === idx;

                return (
                  <g
                    key={bin.label}
                    onMouseEnter={() => setHoveredAgeBin(idx)}
                    onMouseLeave={() => setHoveredAgeBin(null)}
                    className="cursor-crosshair"
                  >
                    <rect
                      x={chartLeft + idx * groupSlot}
                      y={24}
                      width={groupSlot}
                      height={176}
                      fill={isHovered ? '#F3EFE7' : 'transparent'}
                    />
                    <rect
                      x={slotCenter - barWidth - 1}
                      y={200 - lowH}
                      width={barWidth}
                      height={lowH}
                      fill="#1E3A5F"
                    />
                    <rect
                      x={slotCenter + 1}
                      y={200 - highH}
                      width={barWidth}
                      height={highH}
                      fill="#881337"
                    />
                    {ageResolution === '10yr' && (
                      <>
                        <text
                          x={slotCenter - barWidth / 2 - 1}
                          y={Math.max(18, 196 - lowH)}
                          textAnchor="middle"
                          className="fill-[#1E3A5F] text-[9px] font-mono"
                        >
                          {bin.lowRiskCount}
                        </text>
                        <text
                          x={slotCenter + barWidth / 2 + 1}
                          y={Math.max(18, 196 - highH)}
                          textAnchor="middle"
                          className="fill-[#881337] text-[9px] font-mono"
                        >
                          {bin.elevatedRiskCount}
                        </text>
                      </>
                    )}
                    <text
                      x={slotCenter}
                      y={217}
                      textAnchor="middle"
                      className="fill-stone-600 text-[9px] font-mono"
                    >
                      {bin.label}
                    </text>
                  </g>
                );
              })}

              <text
                x={chartLeft + chartWidth / 2}
                y={242}
                textAnchor="middle"
                className="fill-stone-500 text-[10px] font-medium"
              >
                Patient Age Bracket (Years)
              </text>
            </svg>

            <div className="mt-1 h-5 font-mono text-xs tabular-nums text-stone-700">
              {hoveredAgeBin !== null ? (
                (() => {
                  const b = activeAgeBins[hoveredAgeBin];
                  const tot = b.lowRiskCount + b.elevatedRiskCount;
                  const pct = tot > 0 ? ((b.elevatedRiskCount / tot) * 100).toFixed(1) : '0.0';
                  return (
                    <span>
                      Age {b.label}: Low {b.lowRiskCount} · Elevated {b.elevatedRiskCount} (Total{' '}
                      {tot} · Elevated Rate {pct}%)
                    </span>
                  );
                })()
              ) : (
                <span className="font-sans italic text-stone-400">
                  Hover any age bracket to inspect exact counts and class rate.
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Chart 2: Risk Factor Association Comparison */}
        <div className="border border-[#DFD9CE] bg-white p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold text-[#1E3A5F]">
                Section III · Risk Factor Association
              </p>
              <h3 className="mt-0.5 font-display text-lg font-semibold text-stone-900">
                Fig. 3 — Elevated Risk Rate (%) by Clinical Attribute
              </h3>
            </div>
            <div className="flex items-center gap-4 text-xs text-stone-600">
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 bg-[#881337]" /> Present
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 bg-stone-400" /> Absent
              </span>
            </div>
          </div>

          <div className="mt-5 space-y-4">
            {eda.riskFactorComparison.map((item) => (
              <div key={item.factor} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-stone-800">{item.factor}</span>
                  <span className="font-mono tabular-nums text-stone-600">
                    Present: <strong className="text-[#881337]">{item.withFactorPct}%</strong> (n=
                    {item.withCount}) · Absent: {item.withoutFactorPct}% (n={item.withoutCount})
                  </span>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="h-2.5 flex-1 bg-[#ECE7DE]">
                      <div
                        className="h-2.5 bg-[#881337]"
                        style={{ width: `${Math.min(100, item.withFactorPct)}%` }}
                      />
                    </div>
                    <span className="w-11 text-right font-mono text-[11px] font-semibold tabular-nums text-[#881337]">
                      {item.withFactorPct}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-2 flex-1 bg-[#ECE7DE]">
                      <div
                        className="h-2 bg-stone-400"
                        style={{ width: `${Math.min(100, item.withoutFactorPct)}%` }}
                      />
                    </div>
                    <span className="w-11 text-right font-mono text-[11px] tabular-nums text-stone-500">
                      {item.withoutFactorPct}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Chart 3: True Scientific Box-and-Whisker Plot (Wall Thickness & Polyp Size) */}
        <div className="border border-[#DFD9CE] bg-white p-6">
          <p className="text-xs font-semibold text-[#1E3A5F]">
            Section III · Morphological Measurements
          </p>
          <h3 className="mt-0.5 font-display text-lg font-semibold text-stone-900">
            Fig. 4 — Wall Thickness Box-and-Whisker Plot &amp; Polyp Summary
          </h3>
          <p className="mt-1 text-xs text-stone-600">
            Displays exact Min–Max whiskers, Interquartile Box (Q1–Q3), Median line, and Mean (◆)
            on a calibrated 0–13 mm axis.
          </p>

          <div className="mt-4">
            <svg
              viewBox="0 0 540 158"
              className="h-auto w-full"
              role="img"
              aria-label="Gallbladder wall thickness box and whisker plot"
            >
              {[0, 2, 4, 6, 8, 10, 12].map((mm) => {
                const x = 115 + (mm / 13) * 405;
                return (
                  <g key={mm}>
                    <line x1={x} y1={16} x2={x} y2={118} stroke="#ECE7DE" strokeWidth={1} />
                    <text
                      x={x}
                      y={134}
                      textAnchor="middle"
                      className="fill-stone-500 text-[10px] font-mono"
                    >
                      {mm} mm
                    </text>
                  </g>
                );
              })}

              {/* 3.0 mm normal clinical threshold reference */}
              <line
                x1={115 + (3 / 13) * 405}
                y1={16}
                x2={115 + (3 / 13) * 405}
                y2={118}
                stroke="#92400E"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
              <text
                x={115 + (3 / 13) * 405 + 4}
                y={26}
                className="fill-[#92400E] text-[9px] font-mono"
              >
                3.0 mm threshold
              </text>

              {eda.wallThicknessByRisk.map((box, idx) => {
                const cy = idx === 0 ? 48 : 92;
                const scaleX = (v: number) => 115 + (Math.min(13, Math.max(0, v)) / 13) * 405;
                const xMin = scaleX(box.min);
                const xQ1 = scaleX(box.q1);
                const xMed = scaleX(box.median);
                const xMean = scaleX(box.mean);
                const xQ3 = scaleX(box.q3);
                const xMax = scaleX(box.max);
                const strokeCol = idx === 0 ? '#1E3A5F' : '#881337';
                const fillCol = idx === 0 ? '#1E3A5F24' : '#88133724';

                return (
                  <g key={box.label}>
                    <text
                      x={105}
                      y={cy + 3}
                      textAnchor="end"
                      className="fill-stone-800 text-[11px] font-semibold"
                    >
                      {box.label}
                    </text>
                    {/* Whisker horizontal line */}
                    <line x1={xMin} y1={cy} x2={xMax} y2={cy} stroke={strokeCol} strokeWidth={1.5} />
                    {/* Min & Max whisker caps */}
                    <line x1={xMin} y1={cy - 7} x2={xMin} y2={cy + 7} stroke={strokeCol} strokeWidth={1.5} />
                    <line x1={xMax} y1={cy - 7} x2={xMax} y2={cy + 7} stroke={strokeCol} strokeWidth={1.5} />
                    {/* Q1-Q3 IQR Box */}
                    <rect
                      x={xQ1}
                      y={cy - 12}
                      width={Math.max(3, xQ3 - xQ1)}
                      height={24}
                      fill={fillCol}
                      stroke={strokeCol}
                      strokeWidth={1.5}
                    />
                    {/* Median Line */}
                    <line x1={xMed} y1={cy - 12} x2={xMed} y2={cy + 12} stroke={strokeCol} strokeWidth={2.5} />
                    {/* Mean Diamond */}
                    <polygon
                      points={`${xMean},${cy - 5} ${xMean + 5},${cy} ${xMean},${cy + 5} ${xMean - 5},${cy}`}
                      fill="#FFFFFF"
                      stroke={strokeCol}
                      strokeWidth={1.5}
                    />
                  </g>
                );
              })}
            </svg>

            <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2 text-[11px] font-mono tabular-nums text-stone-600">
              {eda.wallThicknessByRisk.map((b) => (
                <div key={b.label} className="border border-[#DFD9CE] bg-[#FAF8F4] px-3 py-2">
                  <span className="font-sans font-semibold text-stone-800">{b.label}:</span> Min{' '}
                  {b.min} · Q1 {b.q1} · Med {b.median} · Mean {b.mean} · Q3 {b.q3} · Max {b.max} mm
                </div>
              ))}
            </div>
          </div>

          <div className="mt-5">
            <h4 className="text-xs font-semibold tracking-wide text-stone-500">
              Table 6. Polyp Size Quartiles (mm)
            </h4>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                    <th className="py-1.5 pl-2 pr-3 font-semibold">Class</th>
                    <th className="py-1.5 px-2 text-right font-semibold">Mean</th>
                    <th className="py-1.5 px-2 text-right font-semibold">Median</th>
                    <th className="py-1.5 px-2 text-right font-semibold">Q3 (75th)</th>
                    <th className="py-1.5 pl-2 pr-2 text-right font-semibold">Max</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#ECE7DE] font-mono tabular-nums">
                  {eda.polypSizeByRisk.map((row) => (
                    <tr key={row.label}>
                      <td className="py-2 pl-2 pr-3 font-sans font-medium text-stone-800">
                        {row.label}
                      </td>
                      <td className="py-2 px-2 text-right">{row.mean} mm</td>
                      <td className="py-2 px-2 text-right">{row.median} mm</td>
                      <td className="py-2 px-2 text-right">{row.q3} mm</td>
                      <td className="py-2 pl-2 pr-2 text-right">{row.max} mm</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Chart 4: Interactive 10x10 Pearson Correlation Heatmap */}
        <div className="border border-[#DFD9CE] bg-white p-6">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-[#1E3A5F]">
                Section III · Multivariate Correlation
              </p>
              <h3 className="mt-0.5 font-display text-lg font-semibold text-stone-900">
                Fig. 5 — Pearson Correlation Matrix (r)
              </h3>
            </div>
            <div className="font-mono text-xs tabular-nums text-stone-600">
              {hoveredCell ? (
                <span>
                  {eda.correlationLabels[hoveredCell.row]} ×{' '}
                  {eda.correlationLabels[hoveredCell.col]} ={' '}
                  <strong>
                    {eda.correlationMatrix[hoveredCell.row][hoveredCell.col].toFixed(2)}
                  </strong>
                </span>
              ) : (
                <span className="font-sans italic text-stone-400">Hover cell for exact r</span>
              )}
            </div>
          </div>

          <div className="mt-4 overflow-x-auto">
            <div className="inline-block min-w-[480px]">
              <div className="grid grid-cols-11 gap-0.5 text-[10px]">
                <div />
                {eda.correlationLabels.map((lbl) => (
                  <div
                    key={`col-${lbl}`}
                    className="truncate px-0.5 py-1 text-center font-medium text-stone-600"
                    title={lbl}
                  >
                    {lbl.slice(0, 6)}
                  </div>
                ))}

                {eda.correlationMatrix.map((rowVals, rIdx) => (
                  <React.Fragment key={`row-${eda.correlationLabels[rIdx]}`}>
                    <div
                      className="flex items-center justify-end truncate pr-2 font-medium text-stone-600"
                      title={eda.correlationLabels[rIdx]}
                    >
                      {eda.correlationLabels[rIdx]}
                    </div>
                    {rowVals.map((val, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        onMouseEnter={() => setHoveredCell({ row: rIdx, col: cIdx })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className="flex h-8 cursor-crosshair items-center justify-center font-mono text-[10px] font-medium tabular-nums transition-transform hover:scale-105"
                        style={{
                          backgroundColor: corrColor(val),
                          color: corrTextColor(val),
                        }}
                      >
                        {val.toFixed(2)}
                      </div>
                    ))}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          {/* Diverging Color Scale Legend */}
          <div className="mt-4 flex items-center justify-between border-t border-[#ECE7DE] pt-3 text-[11px] font-mono text-stone-500">
            <span>−1.0 Inverse</span>
            <div className="flex items-center gap-1">
              {['#881337', '#FECDD3', '#F3EFE7', '#CBD5E1', '#3B5E8C', '#1E3A5F'].map((c) => (
                <span
                  key={c}
                  className="inline-block h-2.5 w-6 border border-[#DFD9CE]"
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
            <span>+1.0 Direct</span>
          </div>
        </div>
      </section>
    </div>
  );
};
