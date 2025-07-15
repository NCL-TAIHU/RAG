// src/components/cards/AppCard.tsx

import "./AppCard.css";

type Props = {
  name: string;
};

export default function AppCard({ name }: Props) {
  return (
    <div className="app-card">
      <h3>{name}</h3>
      <div className="app-card-actions">
        <button>View</button>
        <button>Activate</button>
        <button>Delete</button>
      </div>
    </div>
  );
}
