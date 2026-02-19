---
name: schema-consistency
description: Validate database migrations match SQLAlchemy models, check for NULL columns, orphaned tables, and vocabulary architecture. Use when reviewing migrations, modifying database models, or debugging schema issues with Neon/PostgreSQL.
---

# Schema Consistency

Validates database schema matches code models. Source: `.github/agents/schema-consistency-agent.md`.

## When to Use This Skill

- Reviewing database migrations
- Modifying SQLAlchemy models in `qig-backend/`
- Debugging schema mismatch errors
- Validating pgvector configuration

## Step 1: Check Migration Status

```bash
cd qig-backend
drizzle-kit check
```

## Step 2: Validate Vocabulary Table (CRITICAL)

```sql
-- Only ONE vocabulary table should exist
SELECT table_name FROM information_schema.tables 
WHERE table_name LIKE '%vocab%' OR table_name LIKE '%coordizer%';
```

Expected: Single `coordizer_vocabulary` table.

## Step 3: Check pgvector Configuration

```sql
-- Verify pgvector extension and indexes
SELECT * FROM pg_extension WHERE extname = 'vector';
SELECT indexname FROM pg_indexes WHERE tablename = 'coordizer_vocabulary';
```

## Critical Checks

| Check | Requirement |
|-------|-------------|
| Single vocabulary | Only `coordizer_vocabulary` table |
| Migrations synced | All models have corresponding migrations |
| pgvector indexes | HNSW or IVFFlat indexes on basin_coords |
| No NULL violations | Required columns are NOT NULL |
| No orphans | All tables referenced in code |

## Vocabulary Architecture (CANONICAL)

```python
# ✅ CORRECT: Single canonical vocabulary table
class CoordizerVocabulary(Base):
    __tablename__ = "coordizer_vocabulary"
    token_id: int  # Primary key
    token: str     # The token string
    basin_coords: Vector(64)  # 64D simplex coordinates
    qfi_score: float  # Quantum Fisher Information score
    
# ❌ WRONG: Multiple vocabulary tables
# vocabulary, embeddings, token_embeddings - FORBIDDEN
```

## Validation Commands

```bash
# Generate migration diff
drizzle-kit generate

# Push schema changes
drizzle-kit push

# Check for schema drift
python qig-backend/scripts/validate_schema.py
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA CONSISTENCY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Database Validation:
  - Vocabulary table: ✅ / ❌ (single canonical)
  - Migrations synced: ✅ / ❌
  - pgvector indexes: ✅ / ❌
  - NULL constraints: ✅ / ❌

Schema Issues: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
