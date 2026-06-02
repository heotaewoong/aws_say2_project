#!/bin/bash
# ============================================================
# Smoke Test Script for HAPI FHIR Aurora PostgreSQL Migration
#
# Runs on: fhir-ec2 instance after deployment
# Prerequisites: docker compose up -d already executed
#
# Tests:
#   S1 - Container reaches healthy state within 120 seconds
#   S2 - GET /fhir/metadata returns HTTP 200 with CapabilityStatement
#   S3 - No h2 string in running container environment variables
#
# Validates Requirements: 4.3, 5.1, 1.5, 6.1, 6.2, 6.3
# ============================================================

set -euo pipefail

CONTAINER_NAME="hapi-fhir"
FHIR_BASE="http://localhost:8080"
HEALTH_TIMEOUT=120

PASS_COUNT=0
FAIL_COUNT=0

pass() {
  echo "  [PASS] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  echo "  [FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

echo "============================================"
echo " HAPI FHIR Smoke Tests"
echo "============================================"
echo ""

# ----------------------------------------------------------
# S1: Container reaches healthy state within 120 seconds
# Validates: Requirement 4.3
# ----------------------------------------------------------
echo "S1: Checking container reaches healthy state within ${HEALTH_TIMEOUT}s..."

elapsed=0
healthy=false

while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
  status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not_found")

  if [ "$status" = "healthy" ]; then
    healthy=true
    break
  fi

  sleep 5
  elapsed=$((elapsed + 5))
done

if [ "$healthy" = true ]; then
  pass "S1 - Container is healthy (took ~${elapsed}s)"
else
  fail "S1 - Container did not reach healthy state within ${HEALTH_TIMEOUT}s (status: ${status})"
fi

echo ""

# ----------------------------------------------------------
# S2: GET /fhir/metadata returns HTTP 200 with CapabilityStatement
# Validates: Requirement 5.1
# ----------------------------------------------------------
echo "S2: Checking GET /fhir/metadata returns HTTP 200 with CapabilityStatement..."

http_code=$(curl -s -o /tmp/fhir_metadata.json -w "%{http_code}" "${FHIR_BASE}/fhir/metadata" 2>/dev/null || echo "000")

if [ "$http_code" = "200" ]; then
  resource_type=$(cat /tmp/fhir_metadata.json | grep -o '"resourceType"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | grep -o '"[^"]*"$' | tr -d '"')

  if [ "$resource_type" = "CapabilityStatement" ]; then
    pass "S2 - /fhir/metadata returned HTTP 200 with resourceType: CapabilityStatement"
  else
    fail "S2 - /fhir/metadata returned HTTP 200 but resourceType is '${resource_type}' (expected 'CapabilityStatement')"
  fi
else
  fail "S2 - /fhir/metadata returned HTTP ${http_code} (expected 200)"
fi

rm -f /tmp/fhir_metadata.json

echo ""

# ----------------------------------------------------------
# S3: No h2 string in running container environment variables
# Validates: Requirements 1.5, 6.1, 6.2, 6.3
# ----------------------------------------------------------
echo "S3: Checking no 'h2' string in container environment variables..."

env_output=$(docker exec "$CONTAINER_NAME" env 2>/dev/null || echo "")

if [ -z "$env_output" ]; then
  fail "S3 - Could not retrieve environment variables from container '${CONTAINER_NAME}'"
else
  h2_matches=$(echo "$env_output" | grep -i "h2" || true)

  if [ -z "$h2_matches" ]; then
    pass "S3 - No 'h2' string found in container environment variables"
  else
    fail "S3 - Found 'h2' references in container environment variables:"
    echo "$h2_matches" | while read -r line; do
      echo "       → $line"
    done
  fi
fi

echo ""

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo "============================================"
echo " Summary"
echo "============================================"
echo "  Passed: ${PASS_COUNT}"
echo "  Failed: ${FAIL_COUNT}"
echo "  Total:  $((PASS_COUNT + FAIL_COUNT))"
echo "============================================"

if [ $FAIL_COUNT -gt 0 ]; then
  exit 1
fi

exit 0
