export const MESSAGES = {
  // General
  NO_FILE: "No file selected.",
  UPLOADING: "Uploading file...",
  PROCESSING: "Processing file...",
  SUCCESS: "File processed successfully.",
  PARTIAL_SUCCESS: "File processed, but some fields could not be matched.",
  GENERIC_ERROR: "Something went wrong while processing your file.",
  TRY_AGAIN: "Please check your file and try again.",

  // Frontend validation
  INVALID_TYPE: "Unsupported file type. Only .csv and .json are allowed.",
  EMPTY_FILE: "Selected file is empty.",

  // Network / upload
  SERVER_DOWN: "Cannot connect to server. Please try again in a moment.",
  UPLOAD_FAILED: "Upload failed. Please try again.",

  // Backend file errors
  MISSING_FILE_NAME: "Missing file name.",
  INVALID_FORMAT: "File format could not be understood. Please check your data.",
  INVALID_CSV: "CSV file could not be read. Please check the file format.",
  INVALID_JSON: "JSON file could not be read. Please check the file format.",
  TEXT_DECODE_ERROR: "File could not be decoded as text.",
  FILE_READ_ERROR: "File could not be read by the server.",

  // Data structure errors
  NO_USABLE_DATA: "File has no usable data.",
  NO_USABLE_COLUMNS: "File has no usable columns.",
  DUPLICATE_COLUMNS: "File contains duplicate column names.",
  NON_TABULAR_JSON: "JSON structure is not tabular.",

  // Pipeline errors
  COLUMN_COUNT_MISMATCH:
    "Column processing failed because the column counts do not match.",
  NORMALIZATION_FAILED: "File could not be normalized.",
  UNEXPECTED_ERROR: "Unexpected server error.",

  // Warnings
  UNMAPPED_FIELDS: "Some fields could not be matched.",
};