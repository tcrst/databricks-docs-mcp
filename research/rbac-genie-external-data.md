# External RBAC Data + Genie Per-User Answers

Research notes from Databricks docs MCP server.

## Getting External Data Into Databricks

Several options depending on your data source:

### 1. Direct SQL upload — simplest for small datasets like RBAC user tables

```sql
CREATE TABLE catalog.schema.rbac_users (
  user_email STRING,
  role STRING,
  region STRING,
  business_unit STRING
);
INSERT INTO catalog.schema.rbac_users VALUES
  ('alice@globe.com', 'analyst', 'NCR', 'consumer'),
  ('bob@globe.com', 'manager', 'Visayas', 'b2b');
```

### 2. Lakehouse Federation — live connections to external databases

For PostgreSQL, MySQL, etc.:
- Create a `databricks_connection` resource pointing to your external DB
- Create a `FOREIGN CATALOG` to expose external tables in Unity Catalog
- Data stays in the external system, queried in real-time

### 3. Databricks Apps / Python SDK — programmatic ingestion

Use `databricks-sql-connector` for programmatic data ingestion.

### 4. Terraform — automate user/group creation

```hcl
resource "databricks_user" "unity_users" {
  provider  = databricks.mws
  for_each  = toset(concat(var.databricks_users, var.databricks_metastore_admins))
  user_name = each.key
  force     = true
}

resource "databricks_group" "admin_group" {
  provider     = databricks.mws
  display_name = var.unity_admin_group
}

resource "databricks_group_member" "admin_group_member" {
  provider  = databricks.mws
  for_each  = toset(var.databricks_metastore_admins)
  group_id  = databricks_group.admin_group.id
  member_id = databricks_user.unity_users[each.value].id
}
```

---

## Making Genie Give Different Answers Per User

Two layers work together:

### Layer 1: Unity Catalog Row Filters (Row-Level Security)

Create a SQL function that checks who's querying and filters rows accordingly:

```sql
-- Create a filter function
CREATE FUNCTION catalog.schema.region_filter(region_col STRING)
RETURN is_account_group_member('ncr_team') AND region_col = 'NCR'
    OR is_account_group_member('visayas_team') AND region_col = 'Visayas'
    OR is_account_group_member('admins');  -- admins see everything

-- Apply it to a table
ALTER TABLE catalog.schema.sales_data
SET ROW FILTER catalog.schema.region_filter ON (region);
```

When **Alice** (member of `ncr_team`) asks Genie "show me total sales", she only sees NCR data.
**Bob** (member of `visayas_team`) sees only Visayas data.
**Admins** see everything.

Genie doesn't need to know about this — Unity Catalog enforces it transparently.

### Layer 2: Column Masks (Column-Level Security)

Hide sensitive columns from certain users:

```sql
CREATE FUNCTION catalog.schema.mask_revenue(revenue_col DECIMAL)
RETURN CASE
  WHEN is_account_group_member('finance_team') THEN revenue_col
  ELSE NULL  -- non-finance users can't see revenue figures
END;

ALTER TABLE catalog.schema.sales_data
ALTER COLUMN revenue SET MASK catalog.schema.mask_revenue;
```

### Layer 3: ABAC Policies (Attribute-Based Access Control)

For broader governance — apply policies based on data tags rather than specific tables:

```hcl
resource "databricks_policy_info" "pii_row_filter" {
  on_securable_type     = "CATALOG"
  on_securable_fullname = "main"
  policy_type           = "POLICY_TYPE_ROW_FILTER"
  for_securable_type    = "TABLE"
  to_principals         = ["account users"]
  when_condition        = "hasTag('pii')"
  # ... filter function
}
```

### Key function: `is_account_group_member()`

Returns `TRUE/FALSE` based on whether the current user belongs to a specific group.
Used inside row filters and column masks to branch logic per user/group.

```sql
SELECT is_account_group_member('admins');   -- false
SELECT is_account_group_member('dev');      -- true
```

Related functions:
- `current_user()` — returns the current user's email
- `session_user()` — returns the session user's identity

---

## How It All Connects to Genie

1. **Import your RBAC users/groups** into Databricks (via Terraform, SCIM sync, or manual creation)
2. **Create row filter functions** that use `is_account_group_member()` to check user groups
3. **Apply filters to your tables** in Unity Catalog
4. **Create a single Genie space** pointing to those tables
5. **Genie doesn't need per-user logic** — when User A queries, Unity Catalog automatically filters the data before Genie ever sees it. Different users get different answers from the same Genie space.

> "Data access in a Genie space is governed by Unity Catalog, including any row filters and column masks. Users only see data they have permission to access. Any question about data they can't access generates an empty response."

---

## Google Workspace as RBAC Source

### The Challenge

Globe uses Google Workspace (Google Groups) for organizational RBAC. Databricks needs those groups to enforce row-level security via `is_account_group_member()`. However, **Google Workspace is NOT officially supported** for SCIM provisioning in Databricks.

Databricks officially supports SCIM sync with:
- Azure Active Directory (Entra ID)
- Okta
- OneLogin

Google Workspace is **not** on this list.

### Approach 1: Okta as Identity Bridge (Recommended for production)

Use Okta as middleware between Google Workspace and Databricks:

```
Google Workspace → Okta (SAML/OIDC federation) → Databricks (SCIM sync)
```

- Okta federates with Google Workspace for SSO
- Okta syncs users/groups to Databricks via its native SCIM connector
- Groups in Okta map to Databricks account groups
- `is_account_group_member()` works natively

**Pros:** Officially supported, production-grade, handles deprovisioning
**Cons:** Requires Okta license, adds another system

### Approach 2: Custom Sync Pipeline (Best for VinA SOW3)

Build a lightweight sync service:

```
Google Admin SDK → Python script → Databricks SCIM API
```

#### Google Admin SDK — read groups and members

```python
from googleapiclient.discovery import build

service = build('admin', 'directory_v1', credentials=creds)

# List all groups
groups = service.groups().list(domain='globe.com').execute()

# Get members of a group
members = service.members().list(groupKey='ncr-team@globe.com').execute()
```

#### Databricks Account SCIM API — create/update groups

```bash
# Create a group
POST /api/2.0/accounts/{account_id}/scim/v2/Groups
{
  "displayName": "ncr_team",
  "members": [
    {"value": "<user_id_1>"},
    {"value": "<user_id_2>"}
  ]
}

# Create a user
POST /api/2.0/accounts/{account_id}/scim/v2/Users
{
  "userName": "alice@globe.com",
  "displayName": "Alice",
  "active": true
}
```

#### Terraform automation for the sync infrastructure

```hcl
# Create Databricks groups matching Google Groups
resource "databricks_group" "google_synced" {
  for_each     = var.google_groups_map
  display_name = each.key
}

# Add members based on Google Group membership
resource "databricks_group_member" "synced_members" {
  for_each  = var.group_membership_map
  group_id  = databricks_group.google_synced[each.value.group].id
  member_id = databricks_user.synced_users[each.value.user].id
}
```

**Pros:** No Okta dependency, full control, can run as a scheduled job or Databricks workflow
**Cons:** Custom code to maintain, need to handle edge cases (user deprovisioning, group renames)

### Approach 3: Import as Data Table (Simplest, limited)

Skip identity sync entirely — import Google Groups data as a Databricks table and use `current_user()` instead of `is_account_group_member()`:

```sql
-- Sync Google Groups data into a table
CREATE TABLE catalog.schema.google_groups (
  user_email STRING,
  group_name STRING,
  synced_at TIMESTAMP
);

-- Row filter using current_user() instead of group membership
CREATE FUNCTION catalog.schema.region_filter(region_col STRING)
RETURN EXISTS (
  SELECT 1 FROM catalog.schema.google_groups g
  WHERE g.user_email = current_user()
    AND g.group_name = CASE region_col
      WHEN 'NCR' THEN 'ncr-team@globe.com'
      WHEN 'Visayas' THEN 'visayas-team@globe.com'
    END
) OR EXISTS (
  SELECT 1 FROM catalog.schema.google_groups g
  WHERE g.user_email = current_user()
    AND g.group_name = 'admins@globe.com'
);
```

**Pros:** No identity provider setup, works immediately, uses standard SQL
**Cons:** Doesn't leverage native Databricks group membership, requires table refresh on group changes, `is_account_group_member()` won't work (uses `current_user()` lookups instead)

### Recommended Path for VinA SOW3

**Approach 2 (Custom Sync Pipeline)** is the best fit:

1. Build a Python sync service using Google Admin SDK + Databricks SCIM API
2. Run it as a scheduled Databricks workflow (e.g., every 15 minutes)
3. Maps Google Groups → Databricks account groups 1:1
4. Row filters use native `is_account_group_member()` — no workarounds
5. Can be managed via Terraform for the infrastructure pieces
6. Aligns with EP2 deliverables without requiring Globe to purchase Okta

---

## Unity Catalog Row Filters — Deep Dive

### How Row Filters Work

Row filters are **SQL user-defined functions (UDFs)** that Unity Catalog evaluates transparently on every query. The function must return a `BOOLEAN` — rows where it returns `TRUE` are visible, rows where it returns `FALSE` are hidden.

```sql
CREATE FUNCTION catalog.schema.my_filter(col_param STRING)
RETURN <boolean_expression>;

ALTER TABLE catalog.schema.my_table
SET ROW FILTER catalog.schema.my_filter ON (column_name);
```

### Key Rules

- The filter function must accept parameters matching the column types it filters on
- The function is called **automatically** on every `SELECT`, `UPDATE`, `DELETE` — no opt-out
- Works with **Genie, SQL warehouses, notebooks, JDBC/ODBC** — any Unity Catalog-governed access path
- Multiple columns can be passed to a single filter function
- Only **one row filter** per table (but the function can have complex logic)
- Filter functions can query other tables (e.g., lookup tables for RBAC)
- Applied at the **metastore level** — works across all workspaces sharing the metastore

### Filter Function Patterns

#### Simple group-based filter
```sql
CREATE FUNCTION catalog.schema.bu_filter(bu_col STRING)
RETURN
  CASE
    WHEN is_account_group_member('consumer_team') AND bu_col = 'Consumer' THEN TRUE
    WHEN is_account_group_member('b2b_team') AND bu_col = 'B2B' THEN TRUE
    WHEN is_account_group_member('admins') THEN TRUE
    ELSE FALSE
  END;
```

#### Lookup table filter (for dynamic RBAC)
```sql
CREATE FUNCTION catalog.schema.dynamic_filter(region_col STRING)
RETURN EXISTS (
  SELECT 1 FROM catalog.schema.user_region_access a
  WHERE a.user_email = current_user()
    AND a.region = region_col
    AND a.is_active = TRUE
);
```

#### Multi-column filter
```sql
CREATE FUNCTION catalog.schema.multi_filter(region STRING, bu STRING)
RETURN
  (is_account_group_member('ncr_consumer') AND region = 'NCR' AND bu = 'Consumer')
  OR (is_account_group_member('visayas_b2b') AND region = 'Visayas' AND bu = 'B2B')
  OR is_account_group_member('admins');

ALTER TABLE catalog.schema.sales
SET ROW FILTER catalog.schema.multi_filter ON (region, business_unit);
```

### Managing Row Filters

```sql
-- Apply a filter
ALTER TABLE catalog.schema.my_table
SET ROW FILTER catalog.schema.my_filter ON (column_name);

-- Remove a filter
ALTER TABLE catalog.schema.my_table
DROP ROW FILTER;

-- View existing filters
DESCRIBE EXTENDED catalog.schema.my_table;
-- Look for "Row Filter" in the output
```

### With Terraform

```hcl
resource "databricks_sql_table" "filtered_table" {
  catalog_name = "main"
  schema_name  = "sales"
  name         = "transactions"
  table_type   = "MANAGED"

  column {
    name = "region"
    type = "STRING"
  }

  row_filter {
    function_name = "main.sales.region_filter"
    input_column_names = ["region"]
  }
}
```

### Performance Considerations

- Row filters add overhead to every query — keep filter functions simple
- Filters that query lookup tables benefit from caching (Databricks handles this)
- For large tables, ensure filter columns are part of partition/Z-order strategy
- Filter functions run with the **invoker's permissions**, not the definer's

### Genie + Row Filters Interaction

- Genie generates SQL normally — it has **no awareness** of row filters
- Unity Catalog intercepts the query and applies the filter before execution
- The user sees only their permitted rows in Genie's response
- If a user has no access to any rows, Genie returns an empty result (not an error)
- This is the cleanest integration — zero changes needed in Genie configuration

---

## VinA SOW3 Relevance

This maps directly to **EP2 (RBAC & Policy Service)**:
- Sync Globe's Google Groups → Databricks groups (via custom sync pipeline)
- Apply row filters to scope Genie answers per business unit
- Use `is_account_group_member()` in filter functions
- Single Genie space, different data views per user group
- Custom Python sync service: Google Admin SDK → Databricks SCIM API
- Scheduled as a Databricks workflow for near-real-time group sync
