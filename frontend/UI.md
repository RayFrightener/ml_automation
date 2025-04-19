# Homeowner Loss‑History AI Agent UI Design (Markdown)

## 1. Overview
- **Purpose**: A streamlined, futuristic, human‑in‑the‑loop dashboard for administering the complete loss‑history pipeline. Combines minimalistic design with powerful controls for error monitoring, agent‑driven fixes, and manual overrides.
- **Aesthetic**: Clean lines, open whitespace, light & dark mode toggle, soft neon accents on hover states.

## 2. Main Navigation
- **Sidebar** (left, collapsible) with icon + label and hover tooltip
  1. Home
  2. Drift Monitor
  3. Model Metrics
  4. Overrides
  5. Artifacts
  6. Data Quality
  7. Incidents
  8. Settings

## 3. Home Screen
- **Top Bar**: Date selector, environment (Prod/Dev) toggle, light/dark switch
- **Pipeline Health Card**: Last run timestamp, status icon (✅/⚠️/❌), runtime
- **Error & Fix Chat Widget**
  - Chat interface in bottom right corner
  - Shows agent messages: detected errors, fix proposals, self‑heal confirmations
  - Input box for admin queries: _"Show last error log"_, _"Apply fix #3"_, etc.
- **Quick‑Action Buttons**: Trigger Retrain, Run Self‑Heal, Generate Fix Proposal, Open Code Console

## 4. Drift Monitor
- **Feature Drift Tiles**: Grid cards with current drift % vs. threshold; neon accent on warning
- **Sparkline** for each feature’s drift over time
- **Details Panel**: Full line chart, context info, Self‑Heal / Propose Fix buttons

## 5. Model Metrics
- **Model Selector**: Dropdown or tabs for model1…model5
- **Metric Cards**: RMSE, MSE, MAE, R² with up/down arrows comparing to Prod
- **Charts**
  - RMSE over last N runs (line chart)
  - Actual vs. Predicted scatter (interactive)
  - SHAP summary (embedded image)
- **Download** icons for data or image exports

## 6. Overrides
- **Pending Proposals List**: Table with problem summary, confidence, timestamp
- **Proposal Detail Panel**: Code snippet, drift context, logs
- **Actions**: Approve, Reject, Edit then Approve

## 7. Artifacts
- **S3 Tree View**: Browse `visuals/` folder
- **Image Preview**: SHAP and AVS images
- **Copy S3 Path** & Download actions

## 8. Data Quality
- **Schema Report**: Table of expected vs. actual columns and type mismatches
- **Null & Duplicate Stats**: Small bar charts per column
- **Generate Fix Plan** button triggers chat widget suggestion

## 9. Incidents
- **Open Tickets List**: Status, severity, assignee
- **Create Ticket Form**: Issue summary, severity dropdown, assignee field
- **Sync Button** for external system updates

## 10. Settings
- **Configuration**: Drift thresholds, HyperOpt settings, MLflow experiment name
- **Channel Mapping**: Alert types → Slack channels
- **User Roles**: Admin vs. Viewer permissions for overrides and fixes

## 11. Code Interpreter Console
- **Floating Panel**: Resizable overlay
- **REPL Editor**: Code cell + Run button
- **File Browser**: `/tmp` directory view
- **Output Panel**: Displays stdout, charts
- **History**: List of previous commands for rerun

## 12. Visual Style & UX
- **Typography**: Headings `#`, `##`, `###`; body text in sans-serif
- **Colors**:
  - Dark mode base: `#131313`, accent: `#4ECDC4`
  - Light mode base: `#FFFFFF`, accent: `#1A535C`
- **Components**: 2xl rounded cards, soft shadows, `p-4` spacing
- **Feedback**: Subtle animations, toast notifications

## 13. Integration Points
- **Frontend**: React + Shadcn/UI
- **Charts**: Recharts for drift & metrics
- **API**: OpenAI Assistant endpoints for function calls
- **Realtime**: WebSockets for live updates

## 14. Security & Permissions
- **RBAC**: Role-based access control for Admin vs. Viewer
- **Sandbox**: Code console restricted to `/tmp` only
- **Audit Trail**: Log every manual action and chat command

---
