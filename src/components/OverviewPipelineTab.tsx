import React, { useState } from 'react';
import { Download, RotateCcw } from 'lucide-react';
import { ATTRIBUTE_DICTIONARY } from '../data/pythonProjectFiles';
import { PreprocessingResult, RawPatientRecord } from '../types/medical';

interface OverviewPipelineTabProps {
  sourceLabel: string;
  isSynthetic: boolean;
  datasetSize: number;
  onChangeDatasetSize: (size: number) => void;
  onResetToSynthetic: () => void;
  onDownloadCsv: () => void;
  rawRecords: RawPatientRecord[];
  preprocessing: PreprocessingResult;
  uploadError: string | null;
}

export const OverviewPipelineTab: React.FC<OverviewPipelineTabProps> = ({
  sourceLabel,
  isSynthetic,
  datasetSize,
  onChangeDatasetSize,
  onResetToSynthetic,
  onDownloadCsv,
  rawRecords,
  preprocessing,
  uploadError,
}) => {
  const [tableView, setTableView] = useState<'cleaned' | 'raw'>('cleaned');
  const [schemaExpanded, setSchemaExpanded] = useState(false);

  const previewRows =
    tableView === 'cleaned'
      ? preprocessing.cleanedRecords.slice(0, 10)
      : rawRecords.slice(0, 10);

  const visibleAttributes = schemaExpanded
    ? ATTRIBUTE_DICTIONARY
    : ATTRIBUTE_DICTIONARY.slice(0, 7);

  return (
    <div className="space-y-8">
      {/* Classic Monograph Header */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl space-y-3">
            <p className="text-xs font-semibold tracking-wide text-[#1E3A5F]">
              Section I · Dataset Collection &amp; Preprocessing Pipeline
            </p>
            <h1 className="font-display text-2xl font-bold tracking-tight text-stone-900 text-balance sm:text-3xl">
              Gallbladder Cancer Detection &amp; Risk Prediction Using Big Data Analytics
            </h1>
            <p className="text-sm leading-relaxed text-stone-600">
              An academic Big Data and Machine Learning workbench demonstrating how Apache Spark
              partitioning, leak-free medical preprocessing, and supervised classification models
              analyze gallbladder clinical risk patterns.
            </p>
            <div className="flex flex-wrap items-center gap-2 border-t border-[#ECE7DE] pt-3 text-xs text-stone-500">
              <span className="font-medium text-stone-800">{sourceLabel}</span>
              <span aria-hidden="true">·</span>
              <span>
                {isSynthetic
                  ? 'Synthetic Educational Cohort (non-clinical demonstration data)'
                  : 'External User-Supplied CSV'}
              </span>
              <span aria-hidden="true">·</span>
              <span className="font-mono tabular-nums text-[#881337]">
                Elevated Risk Rate: {preprocessing.positiveRatePct}%
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <div className="flex items-center gap-1 border border-[#DFD9CE] bg-[#F3EFE7] p-1">
              {[2400, 5000, 10000].map((size) => (
                <button
                  key={size}
                  type="button"
                  onClick={() => onChangeDatasetSize(size)}
                  className={`px-3 py-1.5 text-xs font-medium font-mono tabular-nums transition-colors whitespace-nowrap shrink-0 ${
                    isSynthetic && datasetSize === size
                      ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                >
                  {size.toLocaleString()} rows
                </button>
              ))}
            </div>

            {!isSynthetic && (
              <button
                type="button"
                onClick={onResetToSynthetic}
                className="inline-flex items-center gap-1.5 border border-[#DFD9CE] bg-white px-3.5 py-2 text-xs font-medium text-stone-700 hover:bg-[#FAF8F4] whitespace-nowrap shrink-0"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                <span>Use Synthetic Cohort</span>
              </button>
            )}

            <button
              type="button"
              onClick={onDownloadCsv}
              className="inline-flex items-center gap-1.5 border border-[#DFD9CE] bg-white px-3.5 py-2 text-xs font-medium text-stone-700 hover:bg-[#FAF8F4] whitespace-nowrap shrink-0"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {uploadError && (
          <div className="mt-4 border border-[#881337]/30 bg-rose-50 px-4 py-3 text-xs text-[#881337]">
            CSV Upload Error: {uploadError}
          </div>
        )}
      </section>

      {/* 4-Column Classic KPI Ledger */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="border border-[#DFD9CE] bg-white p-5">
          <p className="text-xs font-medium text-stone-500">Raw Ingested Records</p>
          <p className="mt-2 font-mono text-2xl font-semibold tabular-nums text-stone-900">
            {preprocessing.rawRows.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-stone-500">Before deduplication</p>
        </div>

        <div className="border border-[#DFD9CE] bg-white p-5">
          <p className="text-xs font-medium text-stone-500">Duplicate Rows Removed</p>
          <p className="mt-2 font-mono text-2xl font-semibold tabular-nums text-[#92400E]">
            {preprocessing.duplicatesRemoved.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-stone-500">Exact record matches dropped</p>
        </div>

        <div className="border border-[#DFD9CE] bg-white p-5">
          <p className="text-xs font-medium text-stone-500">Missing Values Imputed</p>
          <p className="mt-2 font-mono text-2xl font-semibold tabular-nums text-[#1E3A5F]">
            {preprocessing.totalMissingCells.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-stone-500">Fit strictly on train split</p>
        </div>

        <div className="border border-[#DFD9CE] bg-white p-5">
          <p className="text-xs font-medium text-stone-500">Clean Modeling Cohort</p>
          <p className="mt-2 font-mono text-2xl font-semibold tabular-nums text-stone-900">
            {preprocessing.cleanedRows.toLocaleString()}
          </p>
          <p className="mt-1 font-mono text-xs tabular-nums text-stone-500">
            Train {preprocessing.trainRows.toLocaleString()} · Test{' '}
            {preprocessing.testRows.toLocaleString()}
          </p>
        </div>
      </section>

      {/* Module 2 Imputation Audit & Attribute Dictionary */}
      <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="border border-[#DFD9CE] bg-white p-6 lg:col-span-5">
          <h2 className="font-display text-lg font-semibold text-stone-900">
            Table 1. Leak-Free Imputation &amp; Scaling Audit
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-stone-600">
            To prevent data leakage, medians and modes are computed solely from the training fold (
            {preprocessing.trainRows.toLocaleString()} rows) before transforming the test fold.
          </p>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                  <th className="py-2 pl-2 pr-3 font-semibold">Column</th>
                  <th className="py-2 px-2 text-right font-semibold">Missing</th>
                  <th className="py-2 pl-2 pr-2 font-semibold">Train Imputer</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ECE7DE]">
                {preprocessing.columnMissingStats.map((row) => (
                  <tr key={row.column} className="hover:bg-[#FAF8F4]">
                    <td className="py-2 pl-2 pr-3 font-mono text-stone-800">{row.column}</td>
                    <td className="py-2 px-2 text-right font-mono tabular-nums text-stone-600">
                      {row.missingCount} ({row.missingPct}%)
                    </td>
                    <td className="py-2 pl-2 pr-2 font-mono tabular-nums text-[#1E3A5F]">
                      {row.imputedValue}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="border border-[#DFD9CE] bg-white p-6 lg:col-span-7">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="font-display text-lg font-semibold text-stone-900">
                Table 2. Clinical Feature Dictionary
              </h2>
              <p className="mt-1 text-xs text-stone-600">
                Standardized attributes used across PySpark aggregations and Scikit-learn pipelines.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setSchemaExpanded((prev) => !prev)}
              className="text-xs font-semibold text-[#1E3A5F] underline underline-offset-4 hover:text-[#162C46] whitespace-nowrap shrink-0"
            >
              {schemaExpanded ? 'Show fewer' : `Show all ${ATTRIBUTE_DICTIONARY.length} fields`}
            </button>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                  <th className="py-2 pl-2 pr-3 font-semibold">Attribute</th>
                  <th className="py-2 px-2 font-semibold">Type &amp; Range</th>
                  <th className="py-2 pl-2 pr-2 font-semibold">Clinical Relevance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ECE7DE]">
                {visibleAttributes.map((attr) => (
                  <tr key={attr.name} className="hover:bg-[#FAF8F4]">
                    <td className="py-2 pl-2 pr-3 font-mono font-medium text-stone-900 whitespace-nowrap">
                      {attr.name}
                    </td>
                    <td className="py-2 px-2 text-stone-600 whitespace-nowrap">
                      {attr.type} · {attr.range}
                    </td>
                    <td className="py-2 pl-2 pr-2 text-stone-600">{attr.role}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Before / After Record Table */}
      <section className="border border-[#DFD9CE] bg-white p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-lg font-semibold text-stone-900">
              Table 3. Dataset Record Inspector (First 10 Rows)
            </h2>
            <p className="mt-0.5 text-xs text-stone-600">
              Compare raw records containing missing cells against the cleaned, imputed cohort.
            </p>
          </div>

          <div className="flex items-center gap-1 border border-[#DFD9CE] bg-[#F3EFE7] p-1 self-start">
            <button
              type="button"
              onClick={() => setTableView('cleaned')}
              className={`px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                tableView === 'cleaned'
                  ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              Cleaned Records
            </button>
            <button
              type="button"
              onClick={() => setTableView('raw')}
              className={`px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                tableView === 'raw'
                  ? 'bg-white text-[#1E3A5F] border border-[#DFD9CE]'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              Raw Ingested
            </button>
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                <th className="py-2 pl-2 pr-3 font-semibold">ID</th>
                <th className="py-2 px-2 text-right font-semibold">Age</th>
                <th className="py-2 px-2 font-semibold">Gender</th>
                <th className="py-2 px-2 text-right font-semibold">Gallstones</th>
                <th className="py-2 px-2 text-right font-semibold">Wall (mm)</th>
                <th className="py-2 px-2 text-right font-semibold">Polyp (mm)</th>
                <th className="py-2 px-2 text-right font-semibold">BMI</th>
                <th className="py-2 px-2 text-right font-semibold">Bilirubin</th>
                <th className="py-2 px-2 text-right font-semibold">Alk Phos</th>
                <th className="py-2 pl-2 pr-2 text-right font-semibold">Risk Label</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#ECE7DE] font-mono tabular-nums">
              {previewRows.map((row, idx) => (
                <tr key={`${row.Patient_ID}-${idx}`} className="hover:bg-[#FAF8F4]">
                  <td className="py-2 pl-2 pr-3 text-stone-700">{row.Patient_ID}</td>
                  <td className="py-2 px-2 text-right">
                    {row.Age === null ? <span className="text-[#92400E]">NULL</span> : row.Age}
                  </td>
                  <td className="py-2 px-2 font-sans">
                    {row.Gender === null ? (
                      <span className="font-mono text-[#92400E]">NULL</span>
                    ) : (
                      row.Gender
                    )}
                  </td>
                  <td className="py-2 px-2 text-right">{row.Gallstones ?? 'NULL'}</td>
                  <td className="py-2 px-2 text-right">
                    {row.Wall_Thickness_mm === null ? (
                      <span className="text-[#92400E]">NULL</span>
                    ) : (
                      row.Wall_Thickness_mm.toFixed(2)
                    )}
                  </td>
                  <td className="py-2 px-2 text-right">
                    {row.Polyp_Size_mm !== null ? row.Polyp_Size_mm.toFixed(1) : 'NULL'}
                  </td>
                  <td className="py-2 px-2 text-right">
                    {row.BMI === null ? <span className="text-[#92400E]">NULL</span> : row.BMI.toFixed(1)}
                  </td>
                  <td className="py-2 px-2 text-right">
                    {row.Bilirubin_mg_dL === null ? (
                      <span className="text-[#92400E]">NULL</span>
                    ) : (
                      row.Bilirubin_mg_dL.toFixed(2)
                    )}
                  </td>
                  <td className="py-2 px-2 text-right">{row.Alkaline_Phosphatase_UL ?? 'NULL'}</td>
                  <td
                    className={`py-2 pl-2 pr-2 text-right font-semibold ${
                      row.Cancer_Risk === 1 ? 'text-[#881337]' : 'text-[#14532D]'
                    }`}
                  >
                    {row.Cancer_Risk === 1 ? '1 (Elevated)' : '0 (Low)'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
