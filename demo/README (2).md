/*
  Project: HiveOS Core
  File: validators.js
  Purpose: Task 1 frontend validation helpers.
*/

const SUPPORTED_EXTENSIONS = [".csv", ".json"];

export function validateSelectedFile(file) {
  if (!file) {
    return { isValid: false, message: "No file selected." };
  }

  const lowerName = file.name.toLowerCase();

  const hasSupportedExtension = SUPPORTED_EXTENSIONS.some((ext) =>
    lowerName.endsWith(ext)
  );

  if (!hasSupportedExtension) {
    return {
      isValid: false,
      message: "Unsupported file type. Only .csv and .json are allowed.",
    };
  }

  if (file.size === 0) {
    return {
      isValid: false,
      message: "Selected file is empty.",
    };
  }

  return {
    isValid: true,
    message: "File accepted for Task 1 validation.",
  };
}

export function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "—";
  }

  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function getDisplayFileType(file) {
  const lowerName = file.name.toLowerCase();

  if (lowerName.endsWith(".csv")) {
    return "CSV";
  }

  if (lowerName.endsWith(".json")) {
    return "JSON";
  }

  return "Unknown";
}