/*
  HiveOS Results Dashboard
  tables.css
  Data tables, scroll wrappers, badges, confidence bars, and table states
*/

/* =========================
   TABLE WRAPPERS
   ========================= */

.table-scroll {
  width: 100%;
  max-width: 100%;

  overflow-x: auto;
  overflow-y: hidden;

  border-radius: var(--radius-md);
}

/* =========================
   TABLE BASE
   ========================= */

.data-table,
.simple-table {
  width: 100%;

  border-collapse: separate;
  border-spacing: 0;
}

.data-table {
  min-width: var(--table-min-width);
}

.simple-table {
  min-width: 620px;
}

/* =========================
   TABLE HEADERS
   ========================= */

.data-table thead,
.simple-table thead {
  background: var(--card);
}

.data-table th,
.simple-table th {
  position: sticky;
  top: 0;
  z-index: var(--z-tabs);

  padding: 14px 16px;

  background: var(--card);
  color: var(--muted);

  border-bottom: 1px solid var(--border);

  font-size: var(--text-sm);
  font-weight: var(--fw-semibold);
  text-align: left;
  white-space: nowrap;
}

.data-table th:first-child,
.simple-table th:first-child {
  border-top-left-radius: var(--radius-md);
}

.data-table th:last-child,
.simple-table th:last-child {
  border-top-right-radius: var(--radius-md);
}

.data-table th small {
  display: block;

  margin-top: 4px;

  color: var(--muted);
  font-size: var(--text-xs);
  font-weight: var(--fw-normal);
}

/* =========================
   TABLE BODY
   ========================= */

.data-table tbody tr,
.simple-table tbody tr {
  transition:
    background var(--transition-fast),
    border-color var(--transition-fast);
}

.data-table tbody tr:nth-child(even),
.simple-table tbody tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.018);
}

.data-table tbody tr:hover,
.simple-table tbody tr:hover {
  background: rgba(18, 185, 129, 0.05);
}

.data-table td,
.simple-table td {
  padding: 14px 16px;

  color: var(--text);
  border-bottom: 1px solid var(--border-soft);

  font-size: var(--text-sm);
  vertical-align: middle;
  white-space: nowrap;
}

.data-table tbody tr:last-child td,
.simple-table tbody tr:last-child td {
  border-bottom: none;
}

/* =========================
   BADGES
   ========================= */

.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;

  min-width: 84px;

  padding: 6px 10px;

  border-radius: 999px;

  font-size: var(--text-xs);
  font-weight: var(--fw-semibold);
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.badge.success {
  background: var(--success-soft);
  color: var(--success);
  border: 1px solid rgba(16, 185, 129, 0.28);
}

.badge.warning {
  background: var(--orange-soft);
  color: var(--orange);
  border: 1px solid rgba(245, 158, 11, 0.28);
}

.badge.error {
  background: var(--red-soft);
  color: var(--red);
  border: 1px solid rgba(239, 68, 68, 0.28);
}

.badge.info {
  background: var(--blue-soft);
  color: var(--blue);
  border: 1px solid rgba(37, 99, 235, 0.28);
}

.badge.purple {
  background: var(--purple-soft);
  color: var(--purple);
  border: 1px solid rgba(139, 92, 246, 0.28);
}

/* =========================
   CONFIDENCE BARS
   ========================= */

.confidence-wrap {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.confidence-bar {
  width: 100px;
  height: 8px;

  background: var(--card);
  border-radius: 999px;

  overflow: hidden;
}

.confidence-bar span {
  display: block;
  height: 100%;

  background: var(--primary);
  border-radius: 999px;
}

.confidence-value {
  color: var(--muted);
  font-size: var(--text-xs);
  font-weight: var(--fw-semibold);
}

/* =========================
   EMPTY STATES
   ========================= */

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  padding: 60px 20px;

  text-align: center;
}

.empty-state h3 {
  margin-bottom: var(--space-2);

  font-size: var(--text-lg);
}

.empty-state p {
  max-width: 420px;

  color: var(--muted);
  font-size: var(--text-sm);
}

/* =========================
   TABLE SECTION HEADER
   ========================= */

.table-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);

  margin-bottom: var(--space-4);
}

.table-section-header p {
  color: var(--muted);
  font-size: var(--text-sm);
}

/* =========================
   RESPONSIVE TABLES
   ========================= */

@media (max-width: 820px) {
  .data-table {
    min-width: 760px;
  }

  .simple-table {
    min-width: 620px;
  }

  .data-table th,
  .simple-table th,
  .data-table td,
  .simple-table td {
    padding: 12px;
  }
}