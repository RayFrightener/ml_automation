# Homeowner Loss‑History AI Agent UI Design

## 1. Overview
- **Purpose**: A streamlined, futuristic, human‑in‑the‑loop dashboard for administering the complete loss‑history pipeline. Combines minimalistic design with powerful controls for error monitoring, agent‑driven fixes, and manual overrides.
- **Aesthetic**: Clean lines, open whitespace, a light and dark mode toggle, and soft neon accents on hover states.

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

## Detailed Descriptions and Visual Style Guidelines

### Main Navigation
- **Collapsible Sidebar**: The sidebar should be collapsible to provide more screen space when needed. Each item should have an icon and label, with tooltips appearing on hover to describe the item.
- **Icons and Labels**: Use clear and intuitive icons for each navigation item. Labels should be concise and descriptive.
- **Tooltips**: Tooltips should provide additional information about each navigation item when hovered over.

### Home Screen
- **Top Bar**: The top bar should include a date selector, an environment toggle (Prod/Dev), and a light/dark mode switch. These controls should be easily accessible and intuitive to use.
- **Pipeline Health Card**: This card should display the last run timestamp, a status icon (✅/⚠️/❌), and the runtime. The status icon should change color based on the status (green for ✅, yellow for ⚠️, and red for ❌).
- **Error & Fix Chat Widget**: The chat widget should be located in the bottom right corner of the screen. It should display agent messages, including detected errors, fix proposals, and self-heal confirmations. The input box should allow admin queries such as "Show last error log" or "Apply fix #3".
- **Quick-Action Buttons**: These buttons should be prominently displayed and allow users to trigger retrain, run self-heal, generate fix proposals, and open the code console.

### Drift Monitor
- **Feature Drift Tiles**: Display feature drift tiles in a grid format, showing the current drift percentage and threshold. Use neon accents to highlight warnings.
- **Sparklines**: Include sparklines for each feature’s drift over time to provide a quick visual representation of drift trends.
- **Details Panel**: The details panel should include a full line chart, context information, and buttons for self-heal and propose fix.

### Model Metrics
- **Model Selector**: Implement a model selector using a dropdown or tabs for selecting between model1 to model5.
- **Metric Cards**: Create metric cards for RMSE, MSE, MAE, and R², with up/down arrows indicating comparison to production metrics.
- **Charts**: Include charts for RMSE over the last N runs (line chart), actual vs. predicted scatter (interactive), and SHAP summary (embedded image).
- **Download Icons**: Provide download icons for exporting data or images.

### Overrides
- **Pending Proposals List**: Display a table with pending proposals, including problem summary, confidence, and timestamp.
- **Proposal Detail Panel**: Include a detail panel with code snippets, drift context, and logs.
- **Actions**: Provide actions for approving, rejecting, and editing then approving proposals.

### Artifacts
- **S3 Tree View**: Implement an S3 tree view to browse the `visuals/` folder.
- **Image Preview**: Add image previews for SHAP and AVS images.
- **Copy S3 Path & Download Actions**: Include actions for copying S3 paths and downloading files.

### Data Quality
- **Schema Report**: Display a table of expected vs. actual columns and type mismatches.
- **Null & Duplicate Stats**: Include small bar charts for null and duplicate stats per column.
- **Generate Fix Plan Button**: Add a button to generate a fix plan, which triggers a chat widget suggestion.

### Incidents
- **Open Tickets List**: Display a list of open tickets with status, severity, and assignee.
- **Create Ticket Form**: Include a form for creating new tickets, with fields for issue summary, severity dropdown, and assignee.
- **Sync Button**: Add a sync button for updating external systems.

### Settings
- **Configuration Options**: Provide configuration options for drift thresholds, HyperOpt settings, and MLflow experiment name.
- **Channel Mapping**: Implement channel mapping for alert types to Slack channels.
- **User Roles**: Include user roles for admin vs. viewer permissions for overrides and fixes.

### Code Interpreter Console
- **Floating Panel**: Create a floating panel with a resizable overlay.
- **REPL Editor**: Add a REPL editor with a code cell and run button.
- **File Browser**: Implement a file browser for the `/tmp` directory.
- **Output Panel**: Include an output panel for displaying stdout and charts.
- **History**: Add a history list of previous commands for rerun.

### Visual Style & UX
- **Typography**: Use headings `#`, `##`, `###` for typography and sans-serif for body text.
- **Colors**: Implement dark mode with base color `#131313` and accent `#4ECDC4`, and light mode with base color `#FFFFFF` and accent `#1A535C`.
- **Components**: Use 2xl rounded cards, soft shadows, and `p-4` spacing for components.
- **Feedback**: Add subtle animations and toast notifications for feedback.

### Integration Points
- **Frontend**: Use React + Shadcn/UI for the frontend.
- **Charts**: Implement charts using Recharts for drift and metrics.
- **API**: Use OpenAI Assistant endpoints for function calls.
- **Realtime**: Implement WebSockets for live updates.

### Security & Permissions
- **RBAC**: Implement role-based access control (RBAC) for admin vs. viewer.
- **Sandbox**: Restrict the code console to the `/tmp` directory only.
- **Audit Trail**: Log every manual action and chat command for audit trail.
