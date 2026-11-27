/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo, useState } from 'react';
import { ActiveTab, Navbar } from './components/Navbar';
import { OverviewPipelineTab } from './components/OverviewPipelineTab';
import { BigDataEdaTab } from './components/BigDataEdaTab';
import { MlComparisonTab } from './components/MlComparisonTab';
import { PredictorTab } from './components/PredictorTab';
import { DeliverablesTab } from './components/DeliverablesTab';
import {
  generateSyntheticDataset,
  parseCsvToRecords,
  recordsToCsv,
} from './engine/datasetGenerator';
import { cleanAndPreprocess } from './engine/preprocessing';
import { runSparkPartitionedAnalytics } from './engine/sparkSimulator';
import { computeEdaAnalytics } from './engine/edaAnalytics';
import { trainAllMlModels } from './engine/mlEngine';
import { RawPatientRecord } from './types/medical';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [datasetSize, setDatasetSize] = useState<number>(2400);
  const [uploadedRecords, setUploadedRecords] = useState<RawPatientRecord[] | null>(null);
  const [uploadFilename, setUploadFilename] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [testRatio, setTestRatio] = useState<number>(0.2);
  const [mlSeed, setMlSeed] = useState<number>(42);
  const [sparkPartitions, setSparkPartitions] = useState<number>(4);
  const [sparkRunCounter, setSparkRunCounter] = useState<number>(0);

  const isSynthetic = uploadedRecords === null;

  const rawRecords = useMemo(() => {
    if (uploadedRecords) return uploadedRecords;
    return generateSyntheticDataset(datasetSize, 42, true);
  }, [uploadedRecords, datasetSize]);

  const sourceLabel = isSynthetic
    ? `Synthetic Educational Dataset (${datasetSize.toLocaleString()} generated rows)`
    : `Uploaded CSV: ${uploadFilename ?? 'custom_dataset.csv'}`;

  const preprocessing = useMemo(
    () => cleanAndPreprocess(rawRecords, testRatio, mlSeed),
    [rawRecords, testRatio, mlSeed]
  );

  const sparkResult = useMemo(() => {
    // sparkRunCounter forces a fresh timed MapReduce pass when user clicks Run Spark Aggregation
    void sparkRunCounter;
    return runSparkPartitionedAnalytics(preprocessing.cleanedRecords, sparkPartitions);
  }, [preprocessing.cleanedRecords, sparkPartitions, sparkRunCounter]);

  const eda = useMemo(
    () => computeEdaAnalytics(preprocessing.cleanedRecords),
    [preprocessing.cleanedRecords]
  );

  const trainedModels = useMemo(() => trainAllMlModels(preprocessing), [preprocessing]);

  const handleUploadCsv = async (file: File) => {
    try {
      const text = await file.text();
      const parsed = parseCsvToRecords(text);
      if (parsed.error) {
        setUploadError(parsed.error);
        setActiveTab('overview');
        return;
      }
      setUploadError(null);
      setUploadedRecords(parsed.records);
      setUploadFilename(file.name);
      setActiveTab('overview');
    } catch {
      setUploadError('Unable to read the uploaded CSV file.');
    }
  };

  const handleChangeDatasetSize = (size: number) => {
    setUploadedRecords(null);
    setUploadFilename(null);
    setUploadError(null);
    setDatasetSize(size);
  };

  const handleResetToSynthetic = () => {
    setUploadedRecords(null);
    setUploadFilename(null);
    setUploadError(null);
  };

  const handleDownloadCsv = () => {
    const csvContent = recordsToCsv(rawRecords);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = isSynthetic
      ? `gallbladder_synthetic_${datasetSize}.csv`
      : uploadFilename || 'gallbladder_dataset.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#FAF8F4] text-stone-900 flex flex-col">
      <Navbar activeTab={activeTab} onSelectTab={setActiveTab} onUploadCsv={handleUploadCsv} />

      <main className="mx-auto w-full max-w-[1360px] flex-1 px-6 py-9">
        {activeTab === 'overview' && (
          <OverviewPipelineTab
            sourceLabel={sourceLabel}
            isSynthetic={isSynthetic}
            datasetSize={datasetSize}
            onChangeDatasetSize={handleChangeDatasetSize}
            onResetToSynthetic={handleResetToSynthetic}
            onDownloadCsv={handleDownloadCsv}
            rawRecords={rawRecords}
            preprocessing={preprocessing}
            uploadError={uploadError}
          />
        )}

        {activeTab === 'bigdata_eda' && (
          <BigDataEdaTab
            sparkPartitions={sparkPartitions}
            onChangeSparkPartitions={(p) => {
              setSparkPartitions(p);
              setSparkRunCounter((c) => c + 1);
            }}
            onRunSparkJob={() => setSparkRunCounter((c) => c + 1)}
            sparkResult={sparkResult}
            eda={eda}
          />
        )}

        {activeTab === 'ml_models' && (
          <MlComparisonTab
            models={trainedModels}
            testRatio={testRatio}
            onChangeTestRatio={(r) => setTestRatio(r)}
            onRetrain={() => setMlSeed((s) => s + 7)}
            trainRows={preprocessing.trainRows}
            testRows={preprocessing.testRows}
          />
        )}

        {activeTab === 'predictor' && (
          <PredictorTab models={trainedModels} preprocessing={preprocessing} />
        )}

        {activeTab === 'deliverables' && <DeliverablesTab />}
      </main>

      <footer className="border-t border-[#DFD9CE] bg-white">
        <div className="mx-auto flex max-w-[1360px] flex-col gap-2 px-6 py-4 text-xs text-stone-500 sm:flex-row sm:items-center sm:justify-between">
          <span className="font-serif italic text-stone-700">
            CholeInsight · Gallbladder Cancer Detection &amp; Risk Prediction Using Big Data Analytics
          </span>
          <span>Academic Demonstration Monograph — Not intended for clinical diagnosis.</span>
        </div>
      </footer>
    </div>
  );
}
