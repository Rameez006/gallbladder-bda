import React, { useMemo, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { predictSinglePatient } from '../engine/mlEngine';
import {
  GenderType,
  ModelKey,
  PatientPredictionInput,
  PreprocessingResult,
  TrainedModelSummary,
} from '../types/medical';

interface PredictorTabProps {
  models: Record<ModelKey, TrainedModelSummary>;
  preprocessing: PreprocessingResult;
}

const SAMPLE_PRESETS: Array<{ label: string; values: PatientPredictionInput }> = [
  {
    label: 'Low-Risk Routine Profile',
    values: {
      Age: 38,
      Gender: 'Male',
      Gallstones: 0,
      Wall_Thickness_mm: 2.0,
      Polyp_Size_mm: 0.0,
      Chronic_Cholecystitis: 0,
      Porcelain_Gallbladder: 0,
      Family_History: 0,
      BMI: 23.4,
      Bilirubin_mg_dL: 0.7,
      Alkaline_Phosphatase_UL: 78,
      Jaundice: 0,
      Abdominal_Pain: 0,
    },
  },
  {
    label: 'Moderate Symptomatic Profile',
    values: {
      Age: 54,
      Gender: 'Female',
      Gallstones: 1,
      Wall_Thickness_mm: 3.4,
      Polyp_Size_mm: 4.5,
      Chronic_Cholecystitis: 0,
      Porcelain_Gallbladder: 0,
      Family_History: 0,
      BMI: 28.5,
      Bilirubin_mg_dL: 1.2,
      Alkaline_Phosphatase_UL: 125,
      Jaundice: 0,
      Abdominal_Pain: 1,
    },
  },
  {
    label: 'Elevated-Risk Clinical Profile',
    values: {
      Age: 68,
      Gender: 'Female',
      Gallstones: 1,
      Wall_Thickness_mm: 6.2,
      Polyp_Size_mm: 12.0,
      Chronic_Cholecystitis: 1,
      Porcelain_Gallbladder: 1,
      Family_History: 1,
      BMI: 31.2,
      Bilirubin_mg_dL: 2.8,
      Alkaline_Phosphatase_UL: 230,
      Jaundice: 1,
      Abdominal_Pain: 1,
    },
  },
];

export const PredictorTab: React.FC<PredictorTabProps> = ({ models, preprocessing }) => {
  const [selectedModelKey, setSelectedModelKey] = useState<ModelKey>('random_forest');
  const [formState, setFormState] = useState<PatientPredictionInput>(SAMPLE_PRESETS[1].values);

  const activeModel = models[selectedModelKey] ?? models.random_forest;

  const prediction = useMemo(
    () => predictSinglePatient(activeModel, formState, preprocessing),
    [activeModel, formState, preprocessing]
  );

  const updateField = <K extends keyof PatientPredictionInput>(
    key: K,
    value: PatientPredictionInput[K]
  ) => {
    setFormState((prev) => ({ ...prev, [key]: value }));
  };

  const tierStyles =
    prediction.riskTier === 'Elevated Estimated Risk'
      ? { text: 'text-[#881337]', bg: 'bg-rose-50/70', border: 'border-[#881337]/30', bar: 'bg-[#881337]' }
      : prediction.riskTier === 'Moderate Estimated Risk'
        ? { text: 'text-[#92400E]', bg: 'bg-amber-50/70', border: 'border-[#92400E]/30', bar: 'bg-[#B45309]' }
        : { text: 'text-[#14532D]', bg: 'bg-emerald-50/60', border: 'border-[#14532D]/30', bar: 'bg-[#14532D]' };

  return (
    <div className="space-y-8">
      {/* Mandatory Educational Disclaimer */}
      <section className="border border-[#92400E]/35 bg-amber-50/60 p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#92400E]" aria-hidden="true" />
          <div className="text-xs leading-relaxed text-stone-800">
            <strong className="font-semibold text-[#92400E]">
              Educational Prototype Notice — Not a Medical Diagnostic Tool:
            </strong>{' '}
            Risk scores below reflect statistical patterns learned from educational demonstration
            data. They do not confirm or rule out gallbladder cancer and must never guide clinical
            decisions.
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left: Clinical Input Parameters */}
        <section className="border border-[#DFD9CE] bg-white p-6 lg:col-span-7">
          <div className="flex flex-col gap-3 border-b border-[#ECE7DE] pb-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold text-[#1E3A5F]">
                Section V · Interactive Risk Classification
              </p>
              <h2 className="mt-0.5 font-display text-xl font-bold text-stone-900">
                Sample Patient Clinical Attributes
              </h2>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {SAMPLE_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => setFormState(preset.values)}
                  className="border border-[#DFD9CE] bg-[#F3EFE7] px-2.5 py-1.5 text-[11px] font-medium text-stone-800 hover:bg-[#ECE7DE] whitespace-nowrap shrink-0"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-age">
                Age (years)
              </label>
              <input
                id="input-age"
                type="number"
                min={18}
                max={95}
                value={formState.Age}
                onChange={(e) => updateField('Age', Number(e.target.value) || 20)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 font-mono text-sm tabular-nums text-stone-900"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-gender">
                Biological Sex
              </label>
              <select
                id="input-gender"
                value={formState.Gender}
                onChange={(e) => updateField('Gender', e.target.value as GenderType)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 text-sm text-stone-900"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-bmi">
                BMI (kg/m²)
              </label>
              <input
                id="input-bmi"
                type="number"
                step="0.1"
                min={15}
                max={55}
                value={formState.BMI}
                onChange={(e) => updateField('BMI', Number(e.target.value) || 22)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 font-mono text-sm tabular-nums text-stone-900"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-wall">
                Wall Thickness (mm)
              </label>
              <input
                id="input-wall"
                type="number"
                step="0.1"
                min={1}
                max={15}
                value={formState.Wall_Thickness_mm}
                onChange={(e) => updateField('Wall_Thickness_mm', Number(e.target.value) || 2.0)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 font-mono text-sm tabular-nums text-stone-900"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-polyp">
                Polyp Size (mm)
              </label>
              <input
                id="input-polyp"
                type="number"
                step="0.5"
                min={0}
                max={30}
                value={formState.Polyp_Size_mm}
                onChange={(e) => updateField('Polyp_Size_mm', Number(e.target.value) || 0)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 font-mono text-sm tabular-nums text-stone-900"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-bili">
                Bilirubin (mg/dL)
              </label>
              <input
                id="input-bili"
                type="number"
                step="0.1"
                min={0.1}
                max={15}
                value={formState.Bilirubin_mg_dL}
                onChange={(e) => updateField('Bilirubin_mg_dL', Number(e.target.value) || 0.8)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 font-mono text-sm tabular-nums text-stone-900"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-alp">
                Alkaline Phosphatase (U/L)
              </label>
              <input
                id="input-alp"
                type="number"
                min={20}
                max={600}
                value={formState.Alkaline_Phosphatase_UL}
                onChange={(e) =>
                  updateField('Alkaline_Phosphatase_UL', Number(e.target.value) || 85)
                }
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 font-mono text-sm tabular-nums text-stone-900"
              />
            </div>

            <div className="sm:col-span-2">
              <label className="block text-xs font-medium text-stone-700" htmlFor="input-model">
                Trained Classifier
              </label>
              <select
                id="input-model"
                value={selectedModelKey}
                onChange={(e) => setSelectedModelKey(e.target.value as ModelKey)}
                className="mt-1 w-full border border-[#DFD9CE] bg-white px-3 py-2 text-sm text-stone-900"
              >
                {Object.values(models).map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.name} (Test ROC-AUC: {m.rocAuc.toFixed(3)})
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-5 border-t border-[#ECE7DE] pt-5">
            <p className="text-xs font-semibold text-stone-700">Binary Clinical Indicators</p>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
              {(
                [
                  { key: 'Gallstones', label: 'Gallstones Present' },
                  { key: 'Chronic_Cholecystitis', label: 'Chronic Cholecystitis' },
                  { key: 'Porcelain_Gallbladder', label: 'Porcelain Gallbladder' },
                  { key: 'Family_History', label: 'Family History' },
                  { key: 'Jaundice', label: 'Jaundice' },
                  { key: 'Abdominal_Pain', label: 'Abdominal Pain' },
                ] as const
              ).map((item) => {
                const checked = formState[item.key] === 1;
                return (
                  <label
                    key={item.key}
                    className={`flex cursor-pointer items-center gap-2.5 border px-3 py-2.5 text-xs transition-colors ${
                      checked
                        ? 'border-[#1E3A5F] bg-[#F3EFE7] font-semibold text-[#1E3A5F]'
                        : 'border-[#DFD9CE] bg-white text-stone-700 hover:bg-[#FAF8F4]'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => updateField(item.key, e.target.checked ? 1 : 0)}
                      className="h-4 w-4 accent-[#1E3A5F]"
                    />
                    <span>{item.label}</span>
                  </label>
                );
              })}
            </div>
          </div>
        </section>

        {/* Right: Model Risk Output & Contributing Factors */}
        <section className="border border-[#DFD9CE] bg-white p-6 lg:col-span-5">
          <p className="text-xs font-medium text-stone-500">
            Classifier Output · {prediction.modelName}
          </p>

          <div className={`mt-3 border p-5 ${tierStyles.border} ${tierStyles.bg}`}>
            <p className="text-xs font-medium text-stone-600">Estimated Risk Classification</p>
            <p className={`mt-1 font-display text-2xl font-bold ${tierStyles.text}`}>
              {prediction.riskTier}
            </p>

            <div className="mt-4 flex items-baseline justify-between">
              <span className="text-xs text-stone-600">Calibrated Probability Score</span>
              <span className={`font-mono text-2xl font-semibold tabular-nums ${tierStyles.text}`}>
                {prediction.probabilityPct}%
              </span>
            </div>

            <div className="mt-2 h-2.5 w-full border border-[#DFD9CE] bg-white">
              <div
                className={`h-full transition-all ${tierStyles.bar}`}
                style={{ width: `${Math.min(100, Math.max(2, prediction.probabilityPct))}%` }}
              />
            </div>

            <div className="mt-2 flex justify-between font-mono text-[11px] text-stone-500">
              <span>0% Low</span>
              <span>30% Moderate</span>
              <span>65%+ Elevated</span>
            </div>
          </div>

          <div className="mt-6">
            <h3 className="font-display text-base font-semibold text-stone-900">
              Table 7. Contributing Clinical Factors
            </h3>
            <p className="mt-0.5 text-xs text-stone-500">
              Compares entered attributes against the training cohort average.
            </p>

            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-y border-[#DFD9CE] bg-[#F3EFE7] text-stone-600">
                    <th className="py-2 pl-2 pr-2 font-semibold">Feature</th>
                    <th className="py-2 px-2 text-right font-semibold">Input</th>
                    <th className="py-2 px-2 text-right font-semibold">Cohort Avg</th>
                    <th className="py-2 pl-2 pr-2 text-right font-semibold">Direction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#ECE7DE]">
                  {prediction.contributions.map((c) => (
                    <tr key={c.feature}>
                      <td className="py-2 pl-2 pr-2 font-medium text-stone-800">{c.feature}</td>
                      <td className="py-2 px-2 text-right font-mono tabular-nums text-stone-900">
                        {c.patientValue}
                      </td>
                      <td className="py-2 px-2 text-right font-mono tabular-nums text-stone-500">
                        {c.cohortMean}
                      </td>
                      <td
                        className={`py-2 pl-2 pr-2 text-right font-medium ${
                          c.direction === 'increases'
                            ? 'text-[#881337]'
                            : c.direction === 'decreases'
                              ? 'text-[#14532D]'
                              : 'text-stone-500'
                        }`}
                      >
                        {c.direction === 'increases'
                          ? '▲ Raises score'
                          : c.direction === 'decreases'
                            ? '▼ Lowers score'
                            : '● Neutral'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
