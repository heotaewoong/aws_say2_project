# HAPI FHIR JPA Server — Aurora PostgreSQL Deployment

Deploys HAPI FHIR JPA Server v7.4.0 on **fhir-ec2** (`15.164.21.221`) with Aurora PostgreSQL 16.4 as the persistence backend.

## Prerequisites

Before deploying, ensure the following are in place:

### 1. IAM Role on EC2 (Critical)

The fhir-ec2 instance **must** have an IAM Role attached with the following permission:

```json
{
  "Effect": "Allow",
  "Action": "secretsmanager:GetSecretValue",
  "Resource": "arn:aws:secretsmanager:ap-northeast-2:*:secret:rare-link-ai/aurora/hapi-user*"
}
```

Without this role, the container will fail to start because `entrypoint.sh` cannot retrieve the database password from Secrets Manager.

### 2. Security Group Rules

| Rule | Security Group | Protocol | Port | Source/Destination |
|------|---------------|----------|------|--------------------|
| EC2 inbound (FHIR API) | `sg-03b9bc5d95699b797` | TCP | 8080 | Your IP / ALB |
| EC2 → Aurora | `sg-019a357627f1594db` | TCP | 5432 | EC2 instance |

### 3. Aurora Cluster Running

- **Cluster endpoint:** `patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com:5432`
- **Database:** `rarelink`
- **Schema:** `hapi` (auto-created by Flyway on first startup)
- **User:** `hapi_user`
- **Secret:** `rare-link-ai/aurora/hapi-user` (region: `ap-northeast-2`)

### 4. EC2 Instance

- **Host:** `15.164.21.221` (Ubuntu)
- **User:** `ubuntu`
- **Docker & Docker Compose** installed
- **No SSM agent** — use `scp` or EC2 Instance Connect for file transfer

---

## Deployment Steps

### Step 1: Transfer files to fhir-ec2

```bash
# From your local machine (in the repo root)
scp -r deploy/hapi-fhir ubuntu@15.164.21.221:~/hapi-fhir
```

This copies the following to the EC2 instance:

```
~/hapi-fhir/
├── Dockerfile
├── entrypoint.sh
├── docker-compose.yml
├── README.md
└── tests/
    ├── smoke-test.sh
    └── integration-test.sh
```

### Step 2: SSH into fhir-ec2

```bash
ssh ubuntu@15.164.21.221
```

### Step 3: Build the Docker image

```bash
cd ~/hapi-fhir
docker compose build
```

This builds a custom image based on `hapiproject/hapi:v7.4.0` with AWS CLI and the entrypoint script added.

### Step 4: Start the container

```bash
docker compose up -d
```

The container will:
1. Run `entrypoint.sh` to fetch the DB password from Secrets Manager
2. Start HAPI FHIR JPA Server with PostgreSQL configuration
3. Run Flyway migrations to create the `hapi` schema tables (first run only)

### Step 5: Verify startup

```bash
# Watch logs
docker compose logs -f hapi-fhir

# Check health status (healthy within ~120s)
docker inspect --format='{{.State.Health.Status}}' hapi-fhir
```

Wait until the container reports `healthy` before running tests.

---

## Running Tests

All test scripts run **on the fhir-ec2 instance** after the container is healthy.

### Smoke Tests

Verifies the container started correctly and is responding:

```bash
cd ~/hapi-fhir
chmod +x tests/smoke-test.sh
./tests/smoke-test.sh
```

**What it checks:**
- S1: Container reaches healthy state within 120 seconds
- S2: `GET /fhir/metadata` returns HTTP 200 with `CapabilityStatement`
- S3: No `h2` references in container environment variables

### Integration Tests

Verifies FHIR operations work end-to-end with Aurora:

```bash
chmod +x tests/integration-test.sh
./tests/integration-test.sh
```

**What it checks:**
- I1: Create a Patient resource via POST, then GET it back
- I2: Restart the container, verify the Patient persists (data durability)
- I3: FHIR search returns results (confirms `hfj_resource` table exists in `hapi` schema)

> **Note:** Integration tests will restart the container as part of I2. Allow ~2–3 minutes for the full suite.

---

## Rollback Procedure

If the Aurora deployment fails and you need to revert to the H2 in-memory configuration:

### Step 1: Keep a backup of the H2 compose file

Before deploying, save the original H2-based compose file:

```bash
# On fhir-ec2, before first Aurora deployment
cp docker-compose.yml docker-compose.h2.yml
```

### Step 2: Revert to H2

```bash
cd ~/hapi-fhir

# Stop the Aurora-connected container
docker compose down

# Swap compose files
cp docker-compose.h2.yml docker-compose.yml

# Start with H2 configuration
docker compose up -d
```

### Step 3: Verify rollback

```bash
# Confirm container is running
docker inspect --format='{{.State.Health.Status}}' hapi-fhir

# Check metadata endpoint
curl -f http://localhost:8080/fhir/metadata | head -5
```

> **Warning:** Rolling back to H2 means all data written to Aurora will not be accessible from the FHIR API. H2 is in-memory only — data does not persist across restarts.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Container exits immediately with "Failed to retrieve secret" | IAM Role missing or wrong permissions | Attach IAM Role with `secretsmanager:GetSecretValue` to the EC2 instance |
| Container starts but health check fails | Aurora unreachable (SG rules) | Verify `sg-019a357627f1594db` allows EC2 → Aurora on port 5432 |
| "password authentication failed for user hapi_user" | Secret value doesn't match Aurora password | Update the secret in Secrets Manager or reset the Aurora user password |
| "database rarelink does not exist" | Database not created on Aurora | Connect to Aurora and run `CREATE DATABASE rarelink;` |
| Build fails on `aws` install | Network issue on EC2 | Check EC2 has internet access for downloading AWS CLI |

---

## Architecture Reference

```
fhir-ec2 (15.164.21.221)
│
├── Docker Compose
│   └── hapi-fhir container (port 8080)
│       ├── entrypoint.sh → fetches password from Secrets Manager
│       └── HAPI FHIR JPA v7.4.0 → connects to Aurora via JDBC
│
└── Outbound connections:
    ├── Secrets Manager (ap-northeast-2) — credential retrieval
    └── Aurora PostgreSQL (patient-db-cluster:5432) — data persistence
```
