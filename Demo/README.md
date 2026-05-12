<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HiveOSC</title>

  <!--
    Project: HiveOS Core
    File: index.html
    Purpose: Minimal frontend for file selection and validation.
    Key Constraints:
    - Keep UI simple
    - No graphs or advanced analytics
    - Support CSV and JSON upload flow only
  -->

  <link rel="stylesheet" href="./css/upload/tokens.css" />
  <link rel="stylesheet" href="./css/upload/app.css" />
</head>

<body>
  <main class="wrap">
    <header class="hero">
      <h1 class="title">HiveOS</h1>
      <p class="subtitle">Upload your battery data and see it instantly organized.</p>
    </header>

    <section class="panel controls">
      <h2 class="section-title">Upload a battery data file</h2>

      <label class="file-label" for="fileInput">Choose a CSV or JSON file to get started</label>
      <input
        id="fileInput"
        type="file"
        accept=".csv,.json,application/json,text/csv"
      />

      <p id="statusText" class="status-text">No file selected yet.</p>
    </section>

    <section class="panel instructions">
      <h2 class="section-title">How it works</h2>
      <ol class="instruction-list">
        <li>Upload your battery data file (CSV or JSON).</li>
        <li>HiveOS checks that your file is valid.</li>
        <li>You’ll see a simple breakdown of your data instantly</li>
      </ol>
    </section>

    <section id="resultsPanel" class="panel results-panel hidden" aria-live="polite">
      <h2 class="section-title">File summary</h2>

      <div class="summary-list">
        <div class="summary-row">
          <span class="summary-label">File name</span>
          <span id="fileNameValue" class="summary-value">—</span>
        </div>

        <div class="summary-row">
          <span class="summary-label">File type</span>
          <span id="fileTypeValue" class="summary-value">—</span>
        </div>

        <div class="summary-row">
          <span class="summary-label">File size</span>
          <span id="fileSizeValue" class="summary-value">—</span>
        </div>

        <div class="summary-row">
          <span class="summary-label">Validation</span>
          <span id="validationValue" class="summary-value">—</span>
        </div>
<div class="summary-row">
  <span class="summary-label">Rows / Columns</span>
  <span id="shapeValue" class="summary-value">—</span>
</div>

<div class="summary-row">
  <span class="summary-label">Columns</span>
  <span id="columnsValue" class="summary-value">—</span>
</div>
      </div>
    </section>
  </main>

  <script type="module" src="./js/main.js"></script>
</body>
</html>