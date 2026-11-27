import React, { useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { ModelKey, RocPoint, TrainedModelSummary } from '../types/medical';

interface MlComparisonTabProps {
  models: Record<ModelKey, TrainedModelSummary>;
  testRatio: number;
  onChangeTestRatio: (ratio: number) => void;
  onRetrain: () => void;
  trainRows: number;
  testRows: number;
}

const MODEL_COLORS: Record<ModelKey, string> = {
  logistic_regression: '#1E3A5F',
  decision_tree: '#92400E',
  random_forest: '#14532D',
  svm: '#881337',
};

function interpolateTprAtFpr(curve: RocPoint[], targetFpr: number): number {
  if (curve.length === 0) return 0;
  for (let i = 1; i < curve.length; i++) {
    if (curve[i].fpr >= targetFpr) {
      const prev = curve[i - 1];
      const curr = curve[i];
      const span = curr.fpr - prev.fpr;
      if (span <= 1e-6) return curr.tpr;
      const t = (targetFpr - prev.fpr) / span;
      return prev.tpr + t * (curr.tpr - prev.tpr);
    }
  }
  return curve[curve.length - 1].tpr;
}

export const MlComparisonTab: React.FC<MlComparisonTabProps> = ({
  models,
  testRatio,
  onChangeTestRatio,
  onRetrain,
  trainRows,
  testRows,
}) => {
  const modelList = Object.values(models);
  const bestModel = [...modelList].sort((a, b) => b.rocAuc - a.rocAuc)[0];
  const [selectedKey, setSelectedKey] = useState<ModelKey>(bestModel?.key ?? 'random_forest');
  const [hoverFpr, setHoverFpr] = useState<number | null>(null);
  const [hoveredBar, setHoveredBar] = useState<{ model: string; metric: string; val: string } | null>(
    null
  );

  const selectedModel = models[selectedKey] ?? bestModel;

  const toSvgPath = (points: RocPoint[]) => {
    if (points.length === 0) return '';
    return points
      .map((pt, idx) => {
        const x = 48 + pt.fpr * 376;
        const y = 236 - pt.tpr * 196;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
  };

  const handleRocMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * 460;
    const fpr = Math.min(1, Math.max(0, (svgX - 48) / 376));
    setHoverFpr(Number(fpr.toFixed(3)));
  };

  const metricGroups: Array<{
    key: 'accuracy' | 'precision' | 'recall' | 'f1' | 'rocAuc';
    label: string;
  }> = [
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'precision', label: 'Precision' },
    { key: 'recall', label: 'Recall' },
    { key: 'f1', label: 'F1-Score' },
    { key: 'rocAuc', label: 'ROC-AUC' },
  ];

  return (
    <div className="space-y-8">
      {/* Header & Train-Test Split Control */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-semibold tracking-wide text-[#1E3A5F]">
              Section IV · Supervised Machine Learning Evaluation
            </p>
            <h2 className="mt-1 font-display text-2xl font-bold text-stone-900">
              Multi-Model Training &amp; Held-Out Test Comparison
            </h2>
            <p className="mt-1 text-xs text-stone-600">
              Stratified split ({trainRows.toLocaleString()} train · {testRows.toLocaleString()} test
              records) with Z-score standardization fit strictly inside the training fold.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <div className="flex items-center gap-1 border border-[#DFD9CE] bg-[#F3EFE7] p-1">
              {[
                { label: '70 / 30', ratio: 0.3 },
                { label: '75 / 25', ratio: 0.25 },
                { label: '80 / 20', ratio: 0.2 },
              ].map((opt) => (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => onChangeTestRatio(opt.ratio)}
                  className={`px-3 py-1.5 text-xs font-mono tabular-nums font-medium transition-colors whitespace-nowrap shrink-0 ${
                    Math.abs(testRatio - opt.ratio) < 0.01
                      ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={onRetrain}
              className="inline-flex items-center gap-2 rounded-sm bg-[#1E3A5F] px-4 py-2 text-xs font-semibold text-white hover:bg-[#162C46] whitespace-nowrap shrink-0"
            >
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Re-Train All 4 Models</span>
            </button>
          </div>
        </div>

        {/* Classic Academic Comparison Table */}
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                <th className="py-2.5 pl-2 pr-4 font-semibold">Classifier</th>
                <th className="py-2.5 px-3 text-right font-semibold">Accuracy</th>
                <th className="py-2.5 px-3 text-right font-semibold">Precision</th>
                <th className="py-2.5 px-3 text-right font-semibold">Recall</th>
                <th className="py-2.5 px-3 text-right font-semibold">F1-Score</th>
                <th className="py-2.5 px-3 text-right font-semibold">ROC-AUC</th>
                <th className="py-2.5 pl-3 pr-2 text-right font-semibold">Train Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#ECE7DE] font-mono tabular-nums">
              {modelList.map((m) => {
                const isBest = m.key === bestModel.key;
                const isSelected = m.key === selectedModel.key;
                return (
                  <tr
                    key={m.key}
                    onClick={() => setSelectedKey(m.key)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-[#F3EFE7]/70' : 'hover:bg-[#FAF8F4]'
                    }`}
                  >
                    <td className="py-3 pl-2 pr-4 font-sans font-semibold text-stone-900">
                      <span className="inline-flex items-center gap-2">
                        <span
                          className="inline-block h-2.5 w-2.5"
                          style={{ backgroundColor: MODEL_COLORS[m.key] }}
                        />
                        <span>{m.name}</span>
                        {isBest && (
                          <span className="text-[11px] font-normal italic text-[#1E3A5F]">
                            · Highest ROC-AUC
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-right">{(m.precision * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-right">{(m.recall * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-right">{(m.f1 * 100).toFixed(2)}%</td>
                    <td className="py-3 px-3 text-right font-semibold text-[#1E3A5F]">
                      {m.rocAuc.toFixed(4)}
                    </td>
                    <td className="py-3 pl-3 pr-2 text-right text-stone-500">{m.trainTimeMs} ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ROC Curve & Model Performance Grouped Bar Chart */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* High-Resolution Interactive ROC Curve */}
        <div className="border border-[#DFD9CE] bg-white p-6 lg:col-span-6">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h3 className="font-display text-lg font-semibold text-stone-900">
                Fig. 6 — High-Resolution ROC Curve Comparison
              </h3>
              <p className="mt-1 text-xs text-stone-600">
                Move cursor across the plot to probe exact (FPR, TPR) coordinates for all 4 models.
              </p>
            </div>
          </div>

          <div className="mt-3">
            <svg
              viewBox="0 0 460 284"
              className="h-auto w-full cursor-crosshair"
              role="img"
              aria-label="Interactive ROC Curves"
              onMouseMove={handleRocMouseMove}
              onMouseLeave={() => setHoverFpr(null)}
            >
              {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => {
                const y = 236 - tick * 196;
                const x = 48 + tick * 376;
                return (
                  <g key={tick}>
                    <line x1={48} y1={y} x2={424} y2={y} stroke="#ECE7DE" strokeWidth={1} />
                    <line x1={x} y1={40} x2={x} y2={236} stroke="#F3EFE7" strokeWidth={1} />
                    <text
                      x={41}
                      y={y + 3}
                      textAnchor="end"
                      className="fill-stone-500 text-[10px] font-mono"
                    >
                      {tick.toFixed(1)}
                    </text>
                    <text
                      x={x}
                      y={252}
                      textAnchor="middle"
                      className="fill-stone-500 text-[10px] font-mono"
                    >
                      {tick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Axis borders */}
              <line x1={48} y1={40} x2={48} y2={236} stroke="#D6D3CD" strokeWidth={1} />
              <line x1={48} y1={236} x2={424} y2={236} stroke="#D6D3CD" strokeWidth={1} />

              {/* Random Classifier Diagonal */}
              <line
                x1={48}
                y1={236}
                x2={424}
                y2={40}
                stroke="#A8A29E"
                strokeDasharray="4 4"
                strokeWidth={1.2}
              />

              {modelList.map((m) => (
                <path
                  key={m.key}
                  d={toSvgPath(m.rocCurve)}
                  fill="none"
                  stroke={MODEL_COLORS[m.key]}
                  strokeWidth={m.key === selectedModel.key ? 2.5 : 1.75}
                  opacity={m.key === selectedModel.key ? 1 : 0.78}
                />
              ))}

              {/* Interactive Crosshair Probe */}
              {hoverFpr !== null && (
                <g>
                  <line
                    x1={48 + hoverFpr * 376}
                    y1={40}
                    x2={48 + hoverFpr * 376}
                    y2={236}
                    stroke="#57534E"
                    strokeDasharray="2 2"
                    strokeWidth={1}
                  />
                  {modelList.map((m) => {
                    const tpr = interpolateTprAtFpr(m.rocCurve, hoverFpr);
                    const cx = 48 + hoverFpr * 376;
                    const cy = 236 - tpr * 196;
                    return (
                      <circle
                        key={`pt-${m.key}`}
                        cx={cx}
                        cy={cy}
                        r={3.5}
                        fill={MODEL_COLORS[m.key]}
                        stroke="#FFFFFF"
                        strokeWidth={1.2}
                      />
                    );
                  })}
                </g>
              )}

              <text
                x={236}
                y={273}
                textAnchor="middle"
                className="fill-stone-600 text-[10px] font-medium"
              >
                False Positive Rate (1 − Specificity)
              </text>
            </svg>

            {/* Live Coordinate Readout */}
            <div className="mt-2 min-h-[22px] font-mono text-[11px] tabular-nums text-stone-700">
              {hoverFpr !== null ? (
                <div className="flex flex-wrap gap-x-3 gap-y-1">
                  <span className="font-semibold text-stone-900">
                    FPR {hoverFpr.toFixed(2)}:
                  </span>
                  {modelList.map((m) => (
                    <span key={`read-${m.key}`} style={{ color: MODEL_COLORS[m.key] }}>
                      {m.shortName} TPR {(interpolateTprAtFpr(m.rocCurve, hoverFpr) * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              ) : (
                <div className="flex flex-wrap gap-x-5 gap-y-1">
                  {modelList.map((m) => (
                    <button
                      key={m.key}
                      type="button"
                      onClick={() => setSelectedKey(m.key)}
                      className={`inline-flex items-center gap-1.5 ${
                        m.key === selectedModel.key
                          ? 'font-semibold text-stone-950'
                          : 'text-stone-600'
                      }`}
                    >
                      <span
                        className="inline-block h-2.5 w-2.5"
                        style={{ backgroundColor: MODEL_COLORS[m.key] }}
                      />
                      <span>
                        {m.shortName}: AUC {m.rocAuc.toFixed(4)}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Fig. 7 — Multi-Metric Classifier Comparison Bar Chart */}
        <div className="border border-[#DFD9CE] bg-white p-6 lg:col-span-6">
          <h3 className="font-display text-lg font-semibold text-stone-900">
            Fig. 7 — Classifier Metric Comparison (%)
          </h3>
          <p className="mt-1 text-xs text-stone-600">
            Visual comparison of Accuracy, Precision, Recall, F1-Score, and ROC-AUC across models.
          </p>

          <div className="mt-3">
            <svg
              viewBox="0 0 480 264"
              className="h-auto w-full"
              role="img"
              aria-label="Classifier metric grouped bar chart"
            >
              {[0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((tick) => {
                const y = 216 - ((tick - 0.5) / 0.5) * 184;
                return (
                  <g key={tick}>
                    <line x1={44} y1={y} x2={464} y2={y} stroke="#ECE7DE" strokeWidth={1} />
                    <text
                      x={38}
                      y={y + 3}
                      textAnchor="end"
                      className="fill-stone-500 text-[10px] font-mono"
                    >
                      {Math.round(tick * 100)}%
                    </text>
                  </g>
                );
              })}

              <line x1={44} y1={32} x2={44} y2={216} stroke="#D6D3CD" strokeWidth={1} />
              <line x1={44} y1={216} x2={464} y2={216} stroke="#D6D3CD" strokeWidth={1} />

              {metricGroups.map((grp, gIdx) => {
                const groupLeft = 58 + gIdx * 82;
                return (
                  <g key={grp.key}>
                    {modelList.map((m, mIdx) => {
                      const val = m[grp.key];
                      const clamped = Math.max(0.5, Math.min(1, val));
                      const barH = ((clamped - 0.5) / 0.5) * 184;
                      const x = groupLeft + mIdx * 16;
                      return (
                        <rect
                          key={`${grp.key}-${m.key}`}
                          x={x}
                          y={216 - barH}
                          width={13}
                          height={barH}
                          fill={MODEL_COLORS[m.key]}
                          className="cursor-pointer transition-opacity hover:opacity-80"
                          onMouseEnter={() =>
                            setHoveredBar({
                              model: m.name,
                              metric: grp.label,
                              val: `${(val * 100).toFixed(2)}%`,
                            })
                          }
                          onMouseLeave={() => setHoveredBar(null)}
                        />
                      );
                    })}
                    <text
                      x={groupLeft + 30}
                      y={234}
                      textAnchor="middle"
                      className="fill-stone-700 text-[10px] font-medium"
                    >
                      {grp.label}
                    </text>
                  </g>
                );
              })}
            </svg>

            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex flex-wrap gap-3 text-[11px] text-stone-600">
                {modelList.map((m) => (
                  <span key={`leg-${m.key}`} className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block h-2.5 w-2.5"
                      style={{ backgroundColor: MODEL_COLORS[m.key] }}
                    />
                    {m.shortName}
                  </span>
                ))}
              </div>
              <div className="font-mono text-[11px] tabular-nums text-stone-700">
                {hoveredBar ? (
                  <span>
                    {hoveredBar.model} · {hoveredBar.metric}: <strong>{hoveredBar.val}</strong>
                  </span>
                ) : (
                  <span className="font-sans italic text-stone-400">Hover bar for exact %</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Selected Model Confusion Matrix & Feature Importance */}
      <section className="border border-[#DFD9CE] bg-white p-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="font-display text-lg font-semibold text-stone-900">
              Table 6. Model Diagnostics — Confusion Matrix &amp; Feature Weights
            </h3>
            <p className="text-xs text-stone-600">
              Select any classifier to examine its test-fold confusion matrix and top predictors.
            </p>
          </div>
          <select
            value={selectedModel.key}
            onChange={(e) => setSelectedKey(e.target.value as ModelKey)}
            className="border border-[#DFD9CE] bg-white px-3 py-1.5 text-xs font-medium text-stone-800 focus-visible:outline-2 focus-visible:outline-[#1E3A5F]"
            aria-label="Select model to inspect"
          >
            {modelList.map((m) => (
              <option key={m.key} value={m.key}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-8 lg:grid-cols-12">
          {/* 2x2 Confusion Matrix */}
          <div className="lg:col-span-5">
            <h4 className="text-xs font-semibold tracking-wide text-stone-500">
              Confusion Matrix ({selectedModel.name})
            </h4>
            <div className="mt-2.5 grid grid-cols-3 gap-2 text-center text-xs">
              <div />
              <div className="py-1 font-medium text-stone-500">Pred Low (0)</div>
              <div className="py-1 font-medium text-stone-500">Pred Elevated (1)</div>

              <div className="flex items-center justify-end pr-2 font-medium text-stone-500">
                Actual Low (0)
              </div>
              <div className="border border-[#DFD9CE] bg-[#F3EFE7] p-3">
                <p className="font-mono text-lg font-bold tabular-nums text-[#1E3A5F]">
                  {selectedModel.confusionMatrix.tn}
                </p>
                <p className="text-[11px] text-stone-700">True Negative</p>
              </div>
              <div className="border border-[#DFD9CE] bg-amber-50/70 p-3">
                <p className="font-mono text-lg font-bold tabular-nums text-[#92400E]">
                  {selectedModel.confusionMatrix.fp}
                </p>
                <p className="text-[11px] text-[#92400E]">False Positive</p>
              </div>

              <div className="flex items-center justify-end pr-2 font-medium text-stone-500">
                Actual Elevated (1)
              </div>
              <div className="border border-[#DFD9CE] bg-rose-50/70 p-3">
                <p className="font-mono text-lg font-bold tabular-nums text-[#881337]">
                  {selectedModel.confusionMatrix.fn}
                </p>
                <p className="text-[11px] text-[#881337]">False Negative</p>
              </div>
              <div className="border border-[#DFD9CE] bg-emerald-50/60 p-3">
                <p className="font-mono text-lg font-bold tabular-nums text-[#14532D]">
                  {selectedModel.confusionMatrix.tp}
                </p>
                <p className="text-[11px] text-[#14532D]">True Positive</p>
              </div>
            </div>
          </div>

          {/* Top Feature Importances */}
          <div className="lg:col-span-7">
            <h4 className="text-xs font-semibold tracking-wide text-stone-500">
              Top Contributing Features (%)
            </h4>
            <div className="mt-3 space-y-2.5">
              {selectedModel.featureImportances.slice(0, 8).map((item) => (
                <div key={item.feature} className="space-y-1 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-stone-700">{item.feature}</span>
                    <span className="font-mono tabular-nums text-stone-600">
                      {item.importancePct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-[#ECE7DE]">
                    <div
                      className="h-1.5 bg-[#1E3A5F]"
                      style={{ width: `${Math.min(100, item.importancePct * 2.4)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
