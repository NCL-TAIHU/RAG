// src/pages/Apps.tsx

import { useEffect, useState } from "react";
import RecipeForm from "../components/creation/RecipeForm";
import { listApps } from "../lib/api";
import AppCard from "../components/cards/AppCard";

export default function Apps() {
  const [showForm, setShowForm] = useState(false);
  const [apps, setApps] = useState<string[]>([]);

  useEffect(() => {
    listApps().then(setApps).catch(console.error);
  }, []);

  return (
    <div>
      <div className="header">
        <h1>Apps</h1>
        <button onClick={() => setShowForm(!showForm)}>+ New App</button>
      </div>

      {showForm && (
        <div className="form-wrapper">
          <RecipeForm />
        </div>
      )}

      <div className="card-grid">
        {apps.map((name) => (
          <AppCard key={name} name={name} />
        ))}
      </div>
    </div>
  );
}
