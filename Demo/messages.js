/*
  Project: HiveOS Core
  File: app.css
  Purpose: Simple layout and component styles for the Task 1 frontend.
*/

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

.wrap {
  max-width: 720px;
  margin: 40px auto;
  padding: 0 16px 40px;
}

.hero {
  text-align: center;
  margin-bottom: 20px;
}

.title {
  margin: 0 0 8px;
  font-size: 32px;
  font-weight: 800;
}

.subtitle {
  margin: 0;
  color: var(--muted);
}

.panel {
  background: var(--panel);
  border: 1.5px solid var(--gold);
  border-radius: 12px;
  padding: 20px;
  margin-top: 16px;
}

.section-title {
  margin: 0 0 14px;
  font-size: 18px;
}

.file-label {
  display: block;
  margin-bottom: 10px;
  color: var(--muted);
}

input[type="file"] {
  width: 100%;
  padding: 10px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text);
}

.status-text {
  margin: 12px 0 0;
  color: var(--muted);
}

.instruction-list {
  margin: 0;
  padding-left: 20px;
  color: var(--muted);
  line-height: 1.6;
}

.summary-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.summary-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
}

.summary-label {
  color: var(--muted);
}

.summary-value {
  text-align: right;
  font-weight: 600;
}

.hidden {
  display: none;
}

.status-ok {
  color: var(--primary);
}

.status-error {
  color: var(--danger);
}
