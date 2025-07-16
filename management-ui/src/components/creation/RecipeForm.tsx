// src/components/RecipeForm.tsx

import { useState } from "react";
import DatasetForm from "./forms/DatasetForm";
import VectorSetForm from "./forms/VectorSetForm";
import SearchEngineForm from "./forms/SearchEngineForm";
import RouterForm from "./forms/RouterForm";
import RerankerForm from "./forms/RerankerForm";
import ReviewForm from "./forms/ReviewForm";

import type {
  VectorSetConfig,
  SearchEngineConfig,
  RouterConfig,
  RerankerConfig,
  AppConfig,
} from "../../types/app";

export default function RecipeForm() {
  const [step, setStep] = useState(0);
  const [appName, setAppName] = useState<string>("");
  const [dataset, setDataset] = useState<string>("");
  const [vectorSets, setVectorSets] = useState<VectorSetConfig[]>([]);
  const [searchEngines, setSearchEngines] = useState<SearchEngineConfig[]>([]);
  const [router, setRouter] = useState<RouterConfig>({ type: "simple" });
  const [reranker, setReranker] = useState<RerankerConfig>({ type: "identity" });

  const next = () => setStep((s) => s + 1);
  const back = () => setStep((s) => Math.max(0, s - 1));

  const canProceed = () => {
    if (step === 0) return appName !== "" && dataset !== "";
    if (step === 1) return vectorSets.length > 0;
    if (step === 2) return searchEngines.length > 0;
    return true;
  };

  const AppConfig: AppConfig = {
    name: appName,
    dataset,
    search_engines: searchEngines,
    router,
    reranker,
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "720px", margin: "0 auto" }}>
      <h2 style={{ fontSize: "1.75rem", marginBottom: "1rem" }}>Create New App</h2>

      {step === 0 && (
        <DatasetForm
          appName={appName}
          dataset={dataset}
          onAppNameChange={setAppName}
          onDatasetChange={setDataset}
        />
      )}

      {step === 1 && (
        <VectorSetForm
          dataset={dataset}
          vectorSets={vectorSets}
          onChange={setVectorSets}
        />
      )}

      {step === 2 && (
        <SearchEngineForm
          dataset={dataset}
          vectorSets={vectorSets}
          searchEngines={searchEngines}
          onChange={setSearchEngines}
        />
      )}

      {step === 3 && (
        <RouterForm
          router={router}
          onChange={setRouter}
        />
      )}

      {step === 4 && (
        <RerankerForm
          reranker={reranker}
          onChange={setReranker}
        />
      )}

      {step === 5 && (
        <ReviewForm
          metadata={AppConfig}
          onSubmit={() => {
            console.log("Creating app with metadata:", AppConfig);
            // TODO: send POST request here
          }}
        />
      )}

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "2rem" }}>
        {step > 0 && <button onClick={back}>Back</button>}
        {step < 5 && (
          <button onClick={next} disabled={!canProceed()}>
            Next
          </button>
        )}
      </div>
    </div>
  );
}