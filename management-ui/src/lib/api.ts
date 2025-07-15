import type { AppConfig } from "../types/app";

const BASE_URL = "http://localhost:8001";

export async function listApps(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/model/apps`);
  if (!res.ok) throw new Error("Failed to fetch apps");
  return await res.json();
}

export async function getApp(name: string): Promise<AppConfig> {
  const res = await fetch(`${BASE_URL}/model/app/${name}`);
  if (!res.ok) throw new Error("Failed to fetch app metadata");
  return await res.json();
}

export async function createApp(metadata: AppConfig): Promise<void> {
  const res = await fetch(`${BASE_URL}/model/app`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata),
  });
  if (!res.ok) throw new Error("Failed to create app");
}

export async function activateApp(name: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/model/activate/${name}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to activate app");
}

export async function deleteApp(name: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/model/app/${name}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete app");
}
