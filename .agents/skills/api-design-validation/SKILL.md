---
name: api-design-validation
description: Validate REST API design patterns including endpoint naming, HTTP methods, status codes, request/response schemas, and error handling. Use when designing new endpoints, reviewing API changes, or auditing API consistency. Ensures APIs follow RESTful conventions and project standards.
---

# API Design Validation

Validates REST API design patterns for consistency and RESTful compliance.

## When to Use This Skill

- Designing new API endpoints
- Reviewing API changes in PRs
- Auditing existing API for consistency
- Documenting API contracts
- Debugging API behavior

## Endpoint Naming Conventions

### URL Structure

```
/api/v1/<resource>/<id>/<sub-resource>
```

### Rules

| Rule | Good | Bad |
|------|------|-----|
| Use nouns, not verbs | `/users` | `/getUsers` |
| Use plural nouns | `/users` | `/user` |
| Use kebab-case | `/user-profiles` | `/userProfiles` |
| Use lowercase | `/users` | `/Users` |
| No trailing slash | `/users` | `/users/` |

### Examples

```
GET    /api/v1/users              # List users
POST   /api/v1/users              # Create user
GET    /api/v1/users/:id          # Get user
PUT    /api/v1/users/:id          # Update user
DELETE /api/v1/users/:id          # Delete user
GET    /api/v1/users/:id/posts    # List user's posts
```

## HTTP Methods

| Method | Purpose | Idempotent | Request Body |
|--------|---------|------------|--------------|
| `GET` | Read resource | Yes | No |
| `POST` | Create resource | No | Yes |
| `PUT` | Replace resource | Yes | Yes |
| `PATCH` | Partial update | Yes | Yes |
| `DELETE` | Remove resource | Yes | No |

### Method Selection

```
# Create new resource
POST /api/v1/users

# Full replacement (all fields required)
PUT /api/v1/users/:id

# Partial update (only changed fields)
PATCH /api/v1/users/:id

# Actions on resources (use POST)
POST /api/v1/users/:id/activate
POST /api/v1/users/:id/reset-password
```

## Status Codes

### Success (2xx)

| Code | When to Use |
|------|-------------|
| `200 OK` | Successful GET, PUT, PATCH, DELETE |
| `201 Created` | Successful POST (resource created) |
| `204 No Content` | Successful DELETE (no body) |

### Client Errors (4xx)

| Code | When to Use |
|------|-------------|
| `400 Bad Request` | Invalid request body/params |
| `401 Unauthorized` | Missing/invalid authentication |
| `403 Forbidden` | Authenticated but not authorized |
| `404 Not Found` | Resource doesn't exist |
| `409 Conflict` | Resource conflict (duplicate) |
| `422 Unprocessable Entity` | Validation failed |
| `429 Too Many Requests` | Rate limit exceeded |

### Server Errors (5xx)

| Code | When to Use |
|------|-------------|
| `500 Internal Server Error` | Unexpected server error |
| `502 Bad Gateway` | Upstream service error |
| `503 Service Unavailable` | Server overloaded/maintenance |

## Request/Response Schemas

### Request Body

```json
{
  "data": {
    "type": "user",
    "attributes": {
      "email": "user@example.com",
      "name": "John Doe"
    }
  }
}
```

### Success Response

```json
{
  "data": {
    "id": "123",
    "type": "user",
    "attributes": {
      "email": "user@example.com",
      "name": "John Doe",
      "created_at": "2026-01-29T04:00:00Z"
    }
  },
  "meta": {
    "request_id": "abc-123"
  }
}
```

### List Response

```json
{
  "data": [
    { "id": "1", "type": "user", "attributes": {...} },
    { "id": "2", "type": "user", "attributes": {...} }
  ],
  "meta": {
    "total": 100,
    "page": 1,
    "per_page": 20
  },
  "links": {
    "self": "/api/v1/users?page=1",
    "next": "/api/v1/users?page=2",
    "last": "/api/v1/users?page=5"
  }
}
```

### Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  },
  "meta": {
    "request_id": "abc-123"
  }
}
```

## Validation Checklist

### Endpoint Design

- [ ] Uses plural nouns for resources
- [ ] Uses kebab-case for multi-word resources
- [ ] No verbs in URL (except for actions)
- [ ] Proper nesting for sub-resources
- [ ] Version prefix (`/api/v1/`)

### HTTP Methods

- [ ] GET for read operations
- [ ] POST for create operations
- [ ] PUT/PATCH for update operations
- [ ] DELETE for remove operations
- [ ] POST for actions (non-CRUD)

### Status Codes

- [ ] 200/201 for success
- [ ] 400 for bad request
- [ ] 401 for auth required
- [ ] 403 for forbidden
- [ ] 404 for not found
- [ ] 422 for validation errors
- [ ] 500 for server errors

### Response Format

- [ ] Consistent envelope (`data`, `error`, `meta`)
- [ ] Pagination for list endpoints
- [ ] Request ID in responses
- [ ] ISO 8601 timestamps

## Common Anti-Patterns

### Verbs in URLs

```
# ❌ Bad
GET /api/v1/getUsers
POST /api/v1/createUser
POST /api/v1/deleteUser/:id

# ✅ Good
GET /api/v1/users
POST /api/v1/users
DELETE /api/v1/users/:id
```

### Inconsistent Naming

```
# ❌ Bad (mixed styles)
GET /api/v1/users
GET /api/v1/UserProfiles
GET /api/v1/user_settings

# ✅ Good (consistent kebab-case)
GET /api/v1/users
GET /api/v1/user-profiles
GET /api/v1/user-settings
```

### Wrong Status Codes

```
# ❌ Bad
200 OK with error in body
404 for validation errors
500 for auth failures

# ✅ Good
400/422 for validation errors
401 for auth failures
500 only for server errors
```

## Validation Commands

```bash
# Check route definitions
grep -r "router\." server/routes/ | grep -E "(get|post|put|patch|delete)"

# Find inconsistent naming
grep -rE "/api/v1/[A-Z]" server/  # Should find nothing

# Check for verbs in URLs
grep -rE "/(get|create|update|delete)[A-Z]" server/  # Should find nothing
```
