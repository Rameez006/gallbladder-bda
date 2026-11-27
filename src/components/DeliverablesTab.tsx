import React, { useState } from 'react';
import { Check, Copy, Download } from 'lucide-react';
import { PYTHON_PROJECT_FILES, VIVA_QUESTIONS } from '../data/pythonProjectFiles';

const TEACHER_DEMO_STEPS = [
  {
    step: '01. Introduction & Ethics',
    tab: 'Overview',
    script:
      'State the project title and explain that gallbladder cancer is often diagnosed late. Emphasize that the default 2,400-row cohort is a synthetic educational dataset grounded in clinical literature, and the app also supports uploading external CSV datasets (such as SEER/Kaggle cohorts).',
  },
  {
    step: '02. Leak-Free Preprocessing',
    tab: 'Overview · Table 1',
    script:
      'Point to the Preprocessing KPI strip and Table 1. Explain how exact duplicate rows are removed first, then the data is split into stratified Train and Test sets BEFORE fitting median/mode imputers and Z-score scaling—preventing data leakage.',
  },
  {
    step: '03. Big Data (PySpark)',
    tab: 'Big Data & EDA',
    script:
      'Switch to 8 or 16 partitions and click "Run Spark Aggregation." Explain the Map phase (each partition computes local counts, mean wall thickness, and high-risk filters) and the Reduce phase (merging global groupBy tables for Gallstones and Age Brackets).',
  },
  {
    step: '04. Exploratory Analysis',
    tab: 'Big Data & EDA',
    script:
      'Walk through Fig. 2–5: show how elevated risk rises in older age brackets, how gallstones and chronic cholecystitis increase risk prevalence, how wall thickness quartiles separate the two classes, and how the Pearson heatmap confirms feature correlations.',
  },
  {
    step: '05. ML Model Comparison',
    tab: 'ML Models',
    script:
      'Compare Logistic Regression, Decision Tree, Random Forest, and SVM across Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Explain why Recall and ROC-AUC matter more than plain Accuracy in medical screening, and show the Confusion Matrix.',
  },
  {
    step: '06. Live Prediction & Python Kit',
    tab: 'Risk Predictor & Code',
    script:
      'Switch between the Low-Risk and Elevated-Risk presets to demonstrate real-time probability scoring and contributing factors, then point out the mandatory non-diagnostic disclaimer and the modular Python/PySpark/Streamlit source code in /python_project/.',
  },
];

export const DeliverablesTab: React.FC = () => {
  const [selectedFilePath, setSelectedFilePath] = useState<string>(PYTHON_PROJECT_FILES[1].path);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);
  const [openVivaIdx, setOpenVivaIdx] = useState<number | null>(0);

  const activeFile =
    PYTHON_PROJECT_FILES.find((f) => f.path === selectedFilePath) ?? PYTHON_PROJECT_FILES[0];

  const handleCopy = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedPath(label);
      setTimeout(() => setCopiedPath(null), 1800);
    } catch {
      // Clipboard fallback ignored
    }
  };

  const handleDownloadFile = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename.split('/').pop() || 'file.py';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      {/* Teacher Presentation & Demo Walkthrough Guide */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <p className="text-xs font-semibold tracking-wide text-[#1E3A5F]">
          Presentation Guide · How to Explain This Project to Your Teacher
        </p>
        <h2 className="mt-1 font-display text-2xl font-bold text-stone-900">
          6-Step Live Demonstration &amp; Oral Explanation Script
        </h2>
        <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-stone-600">
          Follow this sequence during your college mini-project evaluation or viva to clearly
          demonstrate every Big Data, Machine Learning, and ethical design decision.
        </p>

        <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {TEACHER_DEMO_STEPS.map((item) => (
            <div key={item.step} className="border border-[#DFD9CE] bg-[#FAF8F4] p-4">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-display font-semibold text-[#1E3A5F]">{item.step}</span>
                <span className="font-mono text-[11px] text-stone-500">{item.tab}</span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-stone-700">{item.script}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Academic Report Summary: Abstract, Problem Statement, Objectives, Methodology */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <p className="text-xs font-semibold tracking-wide text-[#1E3A5F]">
          Section VI · College Mini-Project Documentation
        </p>
        <h2 className="mt-1 font-display text-2xl font-bold text-stone-900">
          Abstract, Problem Statement &amp; Methodology
        </h2>

        <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="space-y-5">
            <div>
              <h3 className="font-display text-base font-semibold text-stone-900">
                1. Project Abstract
              </h3>
              <p className="mt-1.5 text-xs leading-relaxed text-stone-600 first-letter:float-left first-letter:mr-2 first-letter:font-display first-letter:text-3xl first-letter:font-bold first-letter:text-[#1E3A5F]">
                Gallbladder carcinoma is an aggressive hepatobiliary malignancy frequently detected at
                late stages due to nonspecific early symptoms. This mini project demonstrates how Big
                Data Analytics (Apache Spark / PySpark) and supervised Machine Learning (Logistic
                Regression, Decision Tree, Random Forest, and Support Vector Machine) can process
                multi-attribute clinical datasets, uncover statistical risk patterns such as chronic
                gallstones and wall thickening, and provide an interactive educational risk
                prediction interface.
              </p>
            </div>

            <div>
              <h3 className="font-display text-base font-semibold text-stone-900">
                2. Problem Statement
              </h3>
              <p className="mt-1.5 text-xs leading-relaxed text-stone-600">
                Traditional single-node spreadsheet tools struggle to scale across large hospital
                cohorts, while uncalibrated predictive scripts often suffer from data leakage and
                lack transparent risk-factor explanations. There is a clear educational need for an
                end-to-end, leak-free Big Data and ML pipeline that demonstrates distributed
                aggregation, visual exploratory analysis, and reproducible model evaluation.
              </p>
            </div>
          </div>

          <div className="space-y-5">
            <div>
              <h3 className="font-display text-base font-semibold text-stone-900">
                3. Project Objectives
              </h3>
              <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs leading-relaxed text-stone-600">
                <li>
                  Ingest and clean structured gallbladder medical datasets (deduplication, missing
                  value imputation, encoding, and normalization without data leakage).
                </li>
                <li>
                  Demonstrate distributed Big Data transformations and groupBy aggregations via
                  Apache Spark (PySpark).
                </li>
                <li>
                  Compare Logistic Regression, Decision Tree, Random Forest, and SVM using
                  Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
                </li>
                <li>
                  Deliver an ethical, beginner-friendly interactive dashboard with clear synthetic vs
                  real-data distinction.
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-display text-base font-semibold text-stone-900">
                4. Conclusion &amp; Future Scope
              </h3>
              <p className="mt-1.5 text-xs leading-relaxed text-stone-600">
                Combining PySpark aggregations with Scikit-learn Pipelines provides both horizontal
                scalability and strict protection against train-test leakage. Future enhancements
                include multi-modal ultrasound image classification using CNNs, SHAP local
                explainability, MongoDB clinical storage connectors, and Spark MLlib cluster jobs
                on IRB-approved hospital registries.
              </p>
            </div>
          </div>
        </div>

        {/* Clean 6-Step Architecture Pipeline Diagram */}
        <div className="mt-6 border-t border-[#ECE7DE] pt-5">
          <h3 className="font-display text-base font-semibold text-stone-900">
            5. System Architecture Pipeline
          </h3>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {[
              { step: '01', title: 'Data Ingestion', desc: 'Synthetic / CSV dataset' },
              { step: '02', title: 'PySpark Layer', desc: 'Partitioned filter & agg' },
              { step: '03', title: 'Leak-Free Prep', desc: 'Split, impute, Z-scale' },
              { step: '04', title: 'EDA Visuals', desc: 'Age, stones, Pearson r' },
              { step: '05', title: 'ML Training', desc: 'LR, DT, RF, SVM + ROC' },
              { step: '06', title: 'Risk Dashboard', desc: 'Probabilities + Disclaimer' },
            ].map((node) => (
              <div key={node.step} className="border border-[#DFD9CE] bg-[#FAF8F4] p-3">
                <p className="font-mono text-[11px] font-semibold text-[#1E3A5F]">
                  Step {node.step}
                </p>
                <p className="mt-1 text-xs font-semibold text-stone-900">{node.title}</p>
                <p className="mt-0.5 text-[11px] text-stone-500">{node.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Python / PySpark / Streamlit Source Code Explorer & Setup Commands */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <div className="flex flex-col gap-3 border-b border-[#ECE7DE] pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-semibold text-[#1E3A5F]">
              Local Python Package · /python_project/
            </p>
            <h2 className="mt-0.5 font-display text-xl font-bold text-stone-900">
              Complete Python, PySpark &amp; Streamlit Source Kit
            </h2>
            <p className="mt-0.5 text-xs text-stone-600">
              Full Python source files are saved inside the workspace under{' '}
              <code className="font-mono text-stone-800">/python_project/</code>.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <code className="border border-[#DFD9CE] bg-[#F3EFE7] px-3 py-1.5 font-mono text-stone-800">
              pip install -r requirements.txt &amp;&amp; streamlit run app.py
            </code>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="space-y-1.5 lg:col-span-4">
            {PYTHON_PROJECT_FILES.map((file) => {
              const isSelected = file.path === activeFile.path;
              return (
                <button
                  key={file.path}
                  type="button"
                  onClick={() => setSelectedFilePath(file.path)}
                  className={`w-full border px-3.5 py-2.5 text-left transition-colors ${
                    isSelected
                      ? 'border-[#1E3A5F] bg-[#F3EFE7]'
                      : 'border-[#DFD9CE] bg-white hover:bg-[#FAF8F4]'
                  }`}
                >
                  <p className="font-mono text-xs font-semibold text-stone-900">{file.path}</p>
                  <p className="mt-0.5 text-[11px] text-stone-500">{file.moduleLabel}</p>
                </button>
              );
            })}

            <div className="border border-[#DFD9CE] bg-[#FAF8F4] p-3.5 text-xs text-stone-600">
              <p className="font-semibold text-stone-900">Execution Steps</p>
              <ol className="mt-1.5 list-decimal space-y-1 pl-4 font-mono text-[11px]">
                <li>cd python_project</li>
                <li>python -m venv venv</li>
                <li>pip install -r requirements.txt</li>
                <li>streamlit run app.py</li>
              </ol>
            </div>
          </div>

          <div className="border border-[#DFD9CE] bg-[#1C1917] text-stone-100 lg:col-span-8">
            <div className="flex items-center justify-between border-b border-stone-800 px-4 py-2.5">
              <div>
                <p className="font-mono text-xs font-semibold text-white">{activeFile.path}</p>
                <p className="text-[11px] text-stone-400">{activeFile.description}</p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => handleCopy(activeFile.code, activeFile.path)}
                  className="inline-flex items-center gap-1.5 border border-stone-700 bg-stone-900 px-2.5 py-1 text-[11px] font-medium text-stone-200 hover:bg-stone-800 whitespace-nowrap"
                >
                  {copiedPath === activeFile.path ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-400" aria-hidden="true" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" aria-hidden="true" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => handleDownloadFile(activeFile.path, activeFile.code)}
                  className="inline-flex items-center gap-1.5 border border-stone-700 bg-stone-900 px-2.5 py-1 text-[11px] font-medium text-stone-200 hover:bg-stone-800 whitespace-nowrap"
                >
                  <Download className="h-3 w-3" aria-hidden="true" />
                  <span>Download</span>
                </button>
              </div>
            </div>
            <pre className="max-h-[420px] overflow-auto p-4 font-mono text-xs leading-relaxed text-stone-200">
              <code>{activeFile.code}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Viva Questions & Answers */}
      <section className="border border-[#DFD9CE] bg-white p-7">
        <p className="text-xs font-semibold text-[#1E3A5F]">Examination Preparation</p>
        <h2 className="mt-1 font-display text-xl font-bold text-stone-900">
          Viva Questions &amp; Answers (10 Core Topics)
        </h2>

        <div className="mt-4 divide-y divide-[#ECE7DE] border-y border-[#DFD9CE]">
          {VIVA_QUESTIONS.map((item, idx) => {
            const isOpen = openVivaIdx === idx;
            return (
              <div key={item.question}>
                <button
                  type="button"
                  onClick={() => setOpenVivaIdx(isOpen ? null : idx)}
                  className="flex w-full items-center justify-between gap-4 py-3.5 text-left text-sm font-medium text-stone-900 hover:text-[#1E3A5F]"
                >
                  <span>{item.question}</span>
                  <span className="font-mono text-xs text-stone-500 shrink-0">
                    {isOpen ? '−' : '+'}
                  </span>
                </button>
                {isOpen && (
                  <p className="pb-4 text-xs leading-relaxed text-stone-600">{item.answer}</p>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};
