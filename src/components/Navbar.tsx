import React, { useRef } from 'react';
import { Upload } from 'lucide-react';

export type ActiveTab = 'overview' | 'bigdata_eda' | 'ml_models' | 'predictor' | 'deliverables';

interface NavbarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onUploadCsv: (file: File) => void;
}

const NAV_ITEMS: Array<{ id: ActiveTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'bigdata_eda', label: 'Big Data & EDA' },
  { id: 'ml_models', label: 'ML Models' },
  { id: 'predictor', label: 'Risk Predictor' },
  { id: 'deliverables', label: 'Report & Code' },
];

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onSelectTab, onUploadCsv }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadCsv(file);
      e.target.value = '';
    }
  };

  return (
    <header className="sticky top-0 z-30 border-b border-[#DFD9CE] bg-white">
      <div className="mx-auto flex max-w-[1360px] items-center justify-between gap-8 px-6 py-4">
        <button
          type="button"
          onClick={() => onSelectTab('overview')}
          className="font-display text-left text-xl font-bold tracking-tight text-[#1E3A5F] whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#1E3A5F]"
        >
          CholeInsight
        </button>

        <nav className="flex items-center gap-7 overflow-x-auto" aria-label="Primary sections">
          {NAV_ITEMS.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`relative py-1.5 text-sm font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#1E3A5F] ${
                  isActive
                    ? 'text-[#1E3A5F] after:absolute after:inset-x-0 after:-bottom-[17px] after:h-[2px] after:bg-[#1E3A5F]'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="shrink-0">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-2 rounded-sm bg-[#1E3A5F] px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-[#162C46] whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1E3A5F]"
          >
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Upload CSV</span>
          </button>
        </div>
      </div>
    </header>
  );
};
