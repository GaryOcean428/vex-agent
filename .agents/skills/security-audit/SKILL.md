---
name: security-audit
description: Scan for security vulnerabilities including hardcoded secrets, insecure dependencies, SQL injection, XSS, and auth issues. Use when reviewing PRs for security, auditing codebase, setting up CI security gates, or before deployment. Zero tolerance for secrets in code.
---

# Security Audit

Scans for security vulnerabilities and enforces secure coding practices.

## When to Use This Skill

- Reviewing PRs for security issues
- Auditing codebase before deployment
- Setting up CI security gates
- Investigating security incidents
- Onboarding new developers to security standards

## Secret Detection

### Forbidden Patterns

```python
# ❌ NEVER commit these patterns
API_KEY = "sk-..."
SECRET_KEY = "abc123..."
PASSWORD = "hunter2"
DATABASE_URL = "postgres://user:pass@..."
AWS_SECRET_ACCESS_KEY = "..."
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----"
```

### Scan Commands

```bash
# Search for common secret patterns
grep -rE "(api[_-]?key|secret|password|token|credential)s?\s*[=:]\s*['\"][^'\"]+['\"]" \
  --include="*.py" --include="*.ts" --include="*.js" --include="*.env*" .

# Search for AWS keys
grep -rE "AKIA[0-9A-Z]{16}" .

# Search for private keys
grep -rE "-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----" .

# Search for JWT secrets
grep -rE "jwt[_-]?secret" -i .
```

### Environment Variables

```bash
# ✅ Correct: Use environment variables
API_KEY = os.environ.get("API_KEY")
DATABASE_URL = process.env.DATABASE_URL

# ✅ Correct: Use .env files (gitignored)
# .env (NOT committed)
API_KEY=sk-...

# .env.example (committed, no real values)
API_KEY=your-api-key-here
```

### Gitignore Requirements

```gitignore
# Secrets
.env
.env.local
.env.*.local
*.pem
*.key
secrets/
credentials/

# IDE
.idea/
.vscode/settings.json
```

## Dependency Vulnerabilities

### Python

```bash
# Install safety
pip install safety pip-audit

# Scan for vulnerabilities
safety check -r requirements.txt
pip-audit

# Update vulnerable packages
pip install --upgrade <package>
```

### Node.js

```bash
# Built-in audit
npm audit

# Fix vulnerabilities
npm audit fix

# Force fix (may break things)
npm audit fix --force
```

## SQL Injection Prevention

### Forbidden Patterns

```python
# ❌ NEVER use string formatting for queries
query = f"SELECT * FROM users WHERE id = {user_id}"
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(f"DELETE FROM users WHERE id = {id}")
```

### Safe Patterns

```python
# ✅ Use parameterized queries
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# ✅ Use ORM
User.query.filter_by(id=user_id).first()

# ✅ SQLAlchemy with parameters
db.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user_id})
```

### Scan for SQL Injection

```bash
# Find string-formatted queries
grep -rE "(execute|query)\s*\(\s*f['\"]" --include="*.py" .
grep -rE "execute\s*\([^)]*\+[^)]*\)" --include="*.py" .
```

## XSS Prevention

### Forbidden Patterns

```typescript
// ❌ NEVER use dangerouslySetInnerHTML with user input
<div dangerouslySetInnerHTML={{ __html: userInput }} />

// ❌ NEVER use innerHTML with user input
element.innerHTML = userInput;
```

### Safe Patterns

```typescript
// ✅ Use textContent for user input
element.textContent = userInput;

// ✅ Use React's automatic escaping
<div>{userInput}</div>

// ✅ Sanitize if HTML is required
import DOMPurify from 'dompurify';
<div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(userInput) }} />
```

### Scan for XSS

```bash
# Find dangerouslySetInnerHTML
grep -rE "dangerouslySetInnerHTML" --include="*.tsx" --include="*.jsx" .

# Find innerHTML assignments
grep -rE "\.innerHTML\s*=" --include="*.ts" --include="*.js" .
```

## Authentication Security

### Password Handling

```python
# ✅ Use bcrypt or argon2
from passlib.hash import bcrypt
hashed = bcrypt.hash(password)
bcrypt.verify(password, hashed)

# ❌ NEVER store plaintext passwords
# ❌ NEVER use MD5/SHA1 for passwords
# ❌ NEVER use unsalted hashes
```

### JWT Security

```python
# ✅ Use strong secret (256+ bits)
# ✅ Set appropriate expiration
# ✅ Validate all claims

jwt.encode(
    {"sub": user_id, "exp": datetime.utcnow() + timedelta(hours=1)},
    SECRET_KEY,
    algorithm="HS256"
)

# ❌ NEVER use "none" algorithm
# ❌ NEVER skip expiration validation
```

### Session Security

```python
# ✅ Secure cookie settings
session.cookie_secure = True      # HTTPS only
session.cookie_httponly = True    # No JS access
session.cookie_samesite = "Lax"   # CSRF protection
```

## Validation Checklist

### Secrets

- [ ] No hardcoded API keys
- [ ] No hardcoded passwords
- [ ] No private keys in repo
- [ ] .env files gitignored
- [ ] .env.example has no real values

### Dependencies

- [ ] No known vulnerabilities (npm audit, safety)
- [ ] Dependencies up to date
- [ ] Lock files committed

### Injection

- [ ] No string-formatted SQL queries
- [ ] No eval() with user input
- [ ] No innerHTML with user input

### Authentication

- [ ] Passwords hashed with bcrypt/argon2
- [ ] JWT secrets are strong
- [ ] Sessions have secure cookie flags
- [ ] CSRF protection enabled

## CI Security Gate

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Secret scan
        run: |
          grep -rE "(api[_-]?key|secret|password)s?\s*[=:]\s*['\"][^'\"]+['\"]" \
            --include="*.py" --include="*.ts" . && exit 1 || exit 0
      
      - name: npm audit
        run: npm audit --audit-level=high
      
      - name: pip audit
        run: |
          pip install pip-audit
          pip-audit -r requirements.txt
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECURITY AUDIT REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Secrets: ✅ PASS / ❌ FAIL
  - Hardcoded secrets: 0
  - Exposed credentials: 0

Dependencies: ✅ PASS / ❌ FAIL
  - Critical vulnerabilities: 0
  - High vulnerabilities: 0

Injection: ✅ PASS / ❌ FAIL
  - SQL injection risks: 0
  - XSS risks: 0

Authentication: ✅ PASS / ❌ FAIL
  - Weak password hashing: 0
  - Insecure sessions: 0

[If issues found, list each with file:line and fix]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
