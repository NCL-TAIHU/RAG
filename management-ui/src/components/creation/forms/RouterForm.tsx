// src/components/forms/RouterForm.tsx

import type { RouterConfig } from "../../../types/app";

type Props = {
  router: RouterConfig;
  onChange: (router: RouterConfig) => void;
};

export default function RouterForm({ router, onChange }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <label>
        <div>Router Type</div>
        <select
          value={router.type}
          onChange={(e) => onChange({ type: e.target.value as RouterConfig["type"] })}
        >
          <option value="simple">simple</option>
        </select>
      </label>
    </div>
  );
}
