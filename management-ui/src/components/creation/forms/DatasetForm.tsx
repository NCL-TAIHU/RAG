import { AVAILABLE_DATASETS } from "../../../lib/config";

type Props = {
  dataset: string;
  appName: string;
  onAppNameChange: (name: string) => void;
  onDatasetChange: (dataset: string) => void;
};

export default function DatasetForm({ dataset, appName, onAppNameChange, onDatasetChange }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <label>
        <div>App Name</div>
        <input
          type="text"
          value={appName}
          onChange={(e) => onAppNameChange(e.target.value)}
        />
      </label>

      <label>
        <div>Dataset</div>
        <select value={dataset} onChange={(e) => onDatasetChange(e.target.value)}>
          <option value="" disabled>Select a dataset</option>
          {AVAILABLE_DATASETS.map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
