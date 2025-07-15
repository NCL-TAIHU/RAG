import { useState } from "react";

export default function Benchmarks() {
  const [showForm, setShowForm] = useState(false);

  return (
    <main className="main-content">
      <div className="header">
        <h1>Benchmarks</h1>
        <button onClick={() => setShowForm(!showForm)}>+ New Benchmark</button>
      </div>

      {showForm && (
        <div className="form-wrapper">
          {/* You can replace this with <BenchmarkForm /> once implemented */}
          <p>Benchmark creation form goes here.</p>
        </div>
      )}
    </main>
  );
}
