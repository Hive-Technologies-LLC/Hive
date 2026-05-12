/*
  HiveOS Core
  main.js
  Upload page controller

  Purpose:
  - Handle file selection
  - Validate CSV / JSON uploads
  - Send valid file to backend
  - Save backend result to sessionStorage
  - Redirect user to results.html
*/

import { elements } from "./dom.js";
import { MESSAGES } from "./messages.js";
import { validateSelectedFile } from "./validators.js";

const API_BASE_URL = "http://127.0.0.1:8000";

/* =========================
   STATUS HANDLING
   ========================= */

function setStatus(message, isError = false) {
  elements.statusText.textContent = message;

  elements.statusText.classList.remove("status-ok", "status-error");
  elements.statusText.classList.add(isError ? "status-error" : "status-ok");
}

/* =========================
   BACKEND UPLOAD
   ========================= */

async function uploadFileToBackend(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/load-file`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || MESSAGES.UPLOAD_FAILED);
  }

  return data;
}

/* =========================
   ERROR MESSAGES
   ========================= */

function getFriendlyErrorMessage(error) {
  const message = error?.message ?? "";

  if (message.includes("Failed to fetch")) {
    return MESSAGES.SERVER_DOWN;
  }

  if (message.includes("Invalid JSON")) {
    return MESSAGES.INVALID_JSON;
  }

  if (message.includes("Invalid CSV")) {
    return MESSAGES.INVALID_CSV;
  }

  if (message.toLowerCase().includes("empty")) {
    return MESSAGES.EMPTY_FILE;
  }

  return message || MESSAGES.GENERIC_ERROR;
}

/* =========================
   MAIN FILE HANDLER
   ========================= */

async function handleFileChange() {
  const file = elements.fileInput.files?.[0];

  if (!file) {
    setStatus(MESSAGES.NO_FILE, false);
    return;
  }

  const validation = validateSelectedFile(file);

  if (!validation.isValid) {
    setStatus(validation.message, true);
    return;
  }

  setStatus(MESSAGES.UPLOADING, false);

  try {
    const summary = await uploadFileToBackend(file);

    sessionStorage.setItem("hiveosResults", JSON.stringify(summary));
    window.location.href = "./results.html";
  } catch (error) {
    console.error("HiveOS upload failed:", error);
    setStatus(getFriendlyErrorMessage(error), true);
  }
}

/* =========================
   BOOTSTRAP
   ========================= */

elements.fileInput.addEventListener("change", handleFileChange);