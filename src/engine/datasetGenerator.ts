import { GenderType, RawPatientRecord } from '../types/medical';

// Deterministic Mulberry32 PRNG for reproducible educational cohorts
function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalRandom(rng: () => number, mean: number, std: number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * std;
}

function clamp(val: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, val));
}

export function generateSyntheticDataset(
  nSamples = 2400,
  seed = 42,
  injectImperfections = true
): RawPatientRecord[] {
  const rng = createRng(seed);
  const records: RawPatientRecord[] = [];

  for (let i = 0; i < nSamples; i++) {
    const age = Math.round(clamp(normalRandom(rng, 54, 13), 21, 86));
    const gender: GenderType = rng() < 0.64 ? 'Female' : 'Male';

    const gallstoneProb = clamp((gender === 'Female' ? 0.42 : 0.28) + (age - 45) * 0.004, 0.1, 0.85);
    const gallstones = rng() < gallstoneProb ? 1 : 0;

    const chronicProb = clamp(0.12 + 0.35 * gallstones + (age - 50) * 0.003, 0.05, 0.75);
    const chronicCholecystitis = rng() < chronicProb ? 1 : 0;

    const porcelainProb = clamp(0.03 + 0.08 * chronicCholecystitis, 0.01, 0.25);
    const porcelainGallbladder = rng() < porcelainProb ? 1 : 0;

    const familyHistory = rng() < 0.14 ? 1 : 0;
    const bmi = Number(clamp(normalRandom(rng, 26.8 + 2.2 * gallstones, 4.4), 16.5, 45.0).toFixed(1));

    const wallThickness = Number(
      clamp(
        normalRandom(
          rng,
          2.2 + 1.4 * gallstones + 1.8 * chronicCholecystitis + 1.5 * porcelainGallbladder,
          0.9
        ),
        1.0,
        12.5
      ).toFixed(2)
    );

    const hasPolyp = rng() < 0.18;
    const polypSize = hasPolyp
      ? Number(clamp(-Math.log(Math.max(1e-6, rng())) * 5.2 + 2.0, 2.0, 22.0).toFixed(1))
      : 0.0;

    const logit =
      -4.25 +
      0.038 * (age - 50) +
      0.42 * (gender === 'Female' ? 1 : 0) +
      1.35 * gallstones +
      0.52 * (wallThickness - 2.6) +
      0.14 * polypSize +
      0.92 * chronicCholecystitis +
      1.25 * porcelainGallbladder +
      0.85 * familyHistory +
      0.04 * (bmi - 25.0) +
      normalRandom(rng, 0, 1.05);

    const prob = 1 / (1 + Math.exp(-logit));
    const cancerRisk = prob >= 0.5 ? 1 : 0;

    const bilirubin = Number(
      clamp(normalRandom(rng, 0.9 + 0.95 * cancerRisk + 0.55 * gallstones, 0.75), 0.2, 8.5).toFixed(2)
    );
    const alkPhos = Math.round(
      clamp(normalRandom(rng, 92 + 58 * cancerRisk + 32 * gallstones, 42), 35, 480)
    );
    const jaundiceProb = clamp(0.08 + 0.34 * cancerRisk + 0.14 * gallstones, 0.02, 0.85);
    const jaundice = rng() < jaundiceProb ? 1 : 0;
    const painProb = clamp(0.24 + 0.38 * gallstones + 0.22 * cancerRisk, 0.1, 0.92);
    const abdominalPain = rng() < painProb ? 1 : 0;

    records.push({
      Patient_ID: `EDU-${10001 + i}`,
      Age: age,
      Gender: gender,
      Gallstones: gallstones,
      Wall_Thickness_mm: wallThickness,
      Polyp_Size_mm: polypSize,
      Chronic_Cholecystitis: chronicCholecystitis,
      Porcelain_Gallbladder: porcelainGallbladder,
      Family_History: familyHistory,
      BMI: bmi,
      Bilirubin_mg_dL: bilirubin,
      Alkaline_Phosphatase_UL: alkPhos,
      Jaundice: jaundice,
      Abdominal_Pain: abdominalPain,
      Cancer_Risk: cancerRisk,
    });
  }

  if (injectImperfections) {
    const missingRate = 0.016;
    for (let i = 0; i < records.length; i++) {
      if (rng() < missingRate) records[i].Age = null;
      if (rng() < missingRate) records[i].Wall_Thickness_mm = null;
      if (rng() < missingRate) records[i].BMI = null;
      if (rng() < missingRate) records[i].Bilirubin_mg_dL = null;
      if (rng() < missingRate * 0.7) records[i].Gender = null;
    }

    // Append deterministic ~1.5% duplicate records for Module 2 cleaning demonstration
    const dupCount = Math.max(8, Math.round(nSamples * 0.015));
    for (let d = 0; d < dupCount; d++) {
      const idx = Math.floor(rng() * nSamples);
      records.push({ ...records[idx] });
    }
  }

  return records;
}

function parseNumber(val: string | undefined): number | null {
  if (val === undefined) return null;
  const trimmed = val.trim();
  if (!trimmed || trimmed.toLowerCase() === 'na' || trimmed.toLowerCase() === 'nan' || trimmed.toLowerCase() === 'null') {
    return null;
  }
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : null;
}

function parseBinary(val: string | undefined): number | null {
  if (val === undefined) return null;
  const t = val.trim().toLowerCase();
  if (!t || t === 'na' || t === 'nan' || t === 'null') return null;
  if (['1', 'yes', 'true', 'present', 'positive', 'high'].includes(t)) return 1;
  if (['0', 'no', 'false', 'absent', 'negative', 'low'].includes(t)) return 0;
  const num = Number(t);
  if (Number.isFinite(num)) return num > 0 ? 1 : 0;
  return null;
}

function parseGender(val: string | undefined): GenderType | null {
  if (!val) return null;
  const t = val.trim().toLowerCase();
  if (t.startsWith('f') || t === '0') return 'Female';
  if (t.startsWith('m') || t === '1') return 'Male';
  return null;
}

export function parseCsvToRecords(csvText: string): {
  records: RawPatientRecord[];
  error: string | null;
} {
  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 10) {
    return {
      records: [],
      error: 'CSV must contain a header row and at least 10 patient records for train/test splitting.',
    };
  }

  const splitCsvLine = (line: string): string[] => {
    const out: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === ',' && !inQuotes) {
        out.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    out.push(current.trim());
    return out;
  };

  const rawHeaders = splitCsvLine(lines[0]).map((h) =>
    h.toLowerCase().replace(/[^a-z0-9]/g, '')
  );

  const findCol = (aliases: string[]): number => {
    for (const alias of aliases) {
      const idx = rawHeaders.indexOf(alias);
      if (idx !== -1) return idx;
    }
    return -1;
  };

  const idxId = findCol(['patientid', 'id', 'recordid']);
  const idxAge = findCol(['age', 'patientage']);
  const idxGender = findCol(['gender', 'sex']);
  const idxGallstones = findCol(['gallstones', 'cholelithiasis', 'stones']);
  const idxWall = findCol(['wallthicknessmm', 'wallthickness', 'gbwallthickness']);
  const idxPolyp = findCol(['polypsizemm', 'polypsize', 'polyp']);
  const idxChronic = findCol(['chroniccholecystitis', 'cholecystitis']);
  const idxPorcelain = findCol(['porcelaingallbladder', 'porcelain']);
  const idxFamily = findCol(['familyhistory', 'family']);
  const idxBmi = findCol(['bmi', 'bodymassindex']);
  const idxBilirubin = findCol(['bilirubinmgdl', 'bilirubin']);
  const idxAlk = findCol(['alkalinephosphataseul', 'alkalinephosphatase', 'alkphos', 'alp']);
  const idxJaundice = findCol(['jaundice']);
  const idxPain = findCol(['abdominalpain', 'pain']);
  const idxTarget = findCol(['cancerrisk', 'target', 'label', 'diagnosis', 'risk', 'gbc']);

  if (idxAge === -1 && idxGallstones === -1 && idxWall === -1) {
    return {
      records: [],
      error:
        'Could not recognize clinical columns. Expected headers such as Age, Gender, Gallstones, Wall_Thickness_mm, and Cancer_Risk.',
    };
  }

  const records: RawPatientRecord[] = [];
  for (let r = 1; r < lines.length; r++) {
    const cols = splitCsvLine(lines[r]);
    if (cols.length < 3) continue;

    records.push({
      Patient_ID: idxId !== -1 && cols[idxId] ? cols[idxId] : `CSV-${r}`,
      Age: idxAge !== -1 ? parseNumber(cols[idxAge]) : null,
      Gender: idxGender !== -1 ? parseGender(cols[idxGender]) : 'Female',
      Gallstones: idxGallstones !== -1 ? parseBinary(cols[idxGallstones]) : 0,
      Wall_Thickness_mm: idxWall !== -1 ? parseNumber(cols[idxWall]) : null,
      Polyp_Size_mm: idxPolyp !== -1 ? (parseNumber(cols[idxPolyp]) ?? 0) : 0,
      Chronic_Cholecystitis: idxChronic !== -1 ? (parseBinary(cols[idxChronic]) ?? 0) : 0,
      Porcelain_Gallbladder: idxPorcelain !== -1 ? (parseBinary(cols[idxPorcelain]) ?? 0) : 0,
      Family_History: idxFamily !== -1 ? (parseBinary(cols[idxFamily]) ?? 0) : 0,
      BMI: idxBmi !== -1 ? parseNumber(cols[idxBmi]) : null,
      Bilirubin_mg_dL: idxBilirubin !== -1 ? parseNumber(cols[idxBilirubin]) : null,
      Alkaline_Phosphatase_UL: idxAlk !== -1 ? parseNumber(cols[idxAlk]) : null,
      Jaundice: idxJaundice !== -1 ? (parseBinary(cols[idxJaundice]) ?? 0) : 0,
      Abdominal_Pain: idxPain !== -1 ? (parseBinary(cols[idxPain]) ?? 0) : 0,
      Cancer_Risk: idxTarget !== -1 ? parseBinary(cols[idxTarget]) : null,
    });
  }

  if (records.length < 10) {
    return { records: [], error: 'Parsed fewer than 10 valid rows from CSV.' };
  }

  return { records, error: null };
}

export function recordsToCsv(records: RawPatientRecord[]): string {
  const headers = [
    'Patient_ID',
    'Age',
    'Gender',
    'Gallstones',
    'Wall_Thickness_mm',
    'Polyp_Size_mm',
    'Chronic_Cholecystitis',
    'Porcelain_Gallbladder',
    'Family_History',
    'BMI',
    'Bilirubin_mg_dL',
    'Alkaline_Phosphatase_UL',
    'Jaundice',
    'Abdominal_Pain',
    'Cancer_Risk',
  ];

  const rows = records.map((r) =>
    headers
      .map((h) => {
        const val = r[h as keyof RawPatientRecord];
        return val === null || val === undefined ? '' : String(val);
      })
      .join(',')
  );

  return [headers.join(','), ...rows].join('\n');
}
