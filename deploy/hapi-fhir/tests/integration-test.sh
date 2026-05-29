#!/bin/bash
# ============================================================
# Integration Test Script for HAPI FHIR Aurora PostgreSQL Migration
#
# Runs on: fhir-ec2 instance after deployment
# Prerequisites: docker compose up -d already executed, container healthy
#
# Tests:
#   I1 - POST a Patient resource, then GET it back and verify data matches
#   I2 - Restart container, wait for healthy, GET Patient to confirm persistence
#   I3 - Verify FHIR search returns results (confirms hfj_resource table exists)
#
# Validates Requirements: 3.1, 5.2, 5.3, 5.4
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

wait_for_healthy() {
  local elapsed=0

  while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
    status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not_found")

    if [ "$status" = "healthy" ]; then
      return 0
    fi

    sleep 5
    elapsed=$((elapsed + 5))
  done

  return 1
}

echo "============================================"
echo " HAPI FHIR Integration Tests"
echo "============================================"
echo ""

# ----------------------------------------------------------
# I1: POST a Patient resource, then GET it back and verify
# Validates: Requirements 5.2
# ----------------------------------------------------------
echo "I1: POST a Patient resource, then GET it back and verify data matches..."

PATIENT_FAMILY="IntegrationTest"
PATIENT_GIVEN="Aurora"
PATIENT_BIRTHDATE="1990-01-15"

PATIENT_PAYLOAD=$(cat <<EOF
{
  "resourceType": "Patient",
  "name": [
    {
      "family": "${PATIENT_FAMILY}",
      "given": ["${PATIENT_GIVEN}"]
    }
  ],
  "birthDate": "${PATIENT_BIRTHDATE}"
}
EOF
)

# POST the Patient resource
create_response=$(curl -s -w "\n%{http_code}" -X POST "${FHIR_BASE}/fhir/Patient" \
  -H "Content-Type: application/fhir+json" \
  -d "$PATIENT_PAYLOAD" 2>/dev/null || echo -e "\n000")

create_body=$(echo "$create_response" | sed '$d')
create_http_code=$(echo "$create_response" | tail -1)

if [ "$create_http_code" = "201" ]; then
  # Extract the Patient ID from the response
  PATIENT_ID=$(echo "$create_body" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | grep -o '"[^"]*"$' | tr -d '"')

  if [ -z "$PATIENT_ID" ]; then
    fail "I1 - Patient created (HTTP 201) but could not extract ID from response"
  else
    # GET the Patient back
    get_response=$(curl -s -w "\n%{http_code}" "${FHIR_BASE}/fhir/Patient/${PATIENT_ID}" 2>/dev/null || echo -e "\n000")
    get_body=$(echo "$get_response" | sed '$d')
    get_http_code=$(echo "$get_response" | tail -1)

    if [ "$get_http_code" = "200" ]; then
      # Verify the data matches
      got_family=$(echo "$get_body" | grep -o '"family"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | grep -o '"[^"]*"$' | tr -d '"')
      got_given=$(echo "$get_body" | grep -o '"given"[[:space:]]*:[[:space:]]*\["[^"]*"\]' | head -1 | grep -o '"[^"]*"' | tail -1 | tr -d '"')
      got_birthdate=$(echo "$get_body" | grep -o '"birthDate"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | grep -o '"[^"]*"$' | tr -d '"')

      if [ "$got_family" = "$PATIENT_FAMILY" ] && [ "$got_given" = "$PATIENT_GIVEN" ] && [ "$got_birthdate" = "$PATIENT_BIRTHDATE" ]; then
        pass "I1 - Patient created and retrieved successfully (ID: ${PATIENT_ID}, family: ${got_family}, given: ${got_given}, birthDate: ${got_birthdate})"
      else
        fail "I1 - Patient data mismatch. Expected family='${PATIENT_FAMILY}', given='${PATIENT_GIVEN}', birthDate='${PATIENT_BIRTHDATE}'. Got family='${got_family}', given='${got_given}', birthDate='${got_birthdate}'"
      fi
    else
      fail "I1 - GET /fhir/Patient/${PATIENT_ID} returned HTTP ${get_http_code} (expected 200)"
    fi
  fi
else
  fail "I1 - POST /fhir/Patient returned HTTP ${create_http_code} (expected 201)"
fi

echo ""

# ----------------------------------------------------------
# I2: Restart container, wait for healthy, verify data persists
# Validates: Requirements 5.3, 5.4
# ----------------------------------------------------------
echo "I2: Restarting container and verifying data persistence..."

if [ -z "${PATIENT_ID:-}" ]; then
  fail "I2 - Skipped: No Patient ID from I1 (previous test failed)"
else
  # Restart the container
  docker compose restart "$CONTAINER_NAME" 2>/dev/null

  echo "     Waiting for container to become healthy (up to ${HEALTH_TIMEOUT}s)..."

  if wait_for_healthy; then
    # GET the Patient created in I1
    persist_response=$(curl -s -w "\n%{http_code}" "${FHIR_BASE}/fhir/Patient/${PATIENT_ID}" 2>/dev/null || echo -e "\n000")
    persist_body=$(echo "$persist_response" | sed '$d')
    persist_http_code=$(echo "$persist_response" | tail -1)

    if [ "$persist_http_code" = "200" ]; then
      persist_family=$(echo "$persist_body" | grep -o '"family"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | grep -o '"[^"]*"$' | tr -d '"')

      if [ "$persist_family" = "$PATIENT_FAMILY" ]; then
        pass "I2 - Patient (ID: ${PATIENT_ID}) persisted after container restart (family: ${persist_family})"
      else
        fail "I2 - Patient data changed after restart. Expected family='${PATIENT_FAMILY}', got '${persist_family}'"
      fi
    else
      fail "I2 - GET /fhir/Patient/${PATIENT_ID} returned HTTP ${persist_http_code} after restart (expected 200)"
    fi
  else
    fail "I2 - Container did not reach healthy state within ${HEALTH_TIMEOUT}s after restart"
  fi
fi

echo ""

# ----------------------------------------------------------
# I3: Verify FHIR search returns results (confirms hfj_resource table)
# Validates: Requirement 3.1
# ----------------------------------------------------------
echo "I3: Verifying FHIR search returns results (confirms hapi schema tables exist)..."

search_response=$(curl -s -w "\n%{http_code}" "${FHIR_BASE}/fhir/Patient?family=${PATIENT_FAMILY}" 2>/dev/null || echo -e "\n000")
search_body=$(echo "$search_response" | sed '$d')
search_http_code=$(echo "$search_response" | tail -1)

if [ "$search_http_code" = "200" ]; then
  # Check that the response is a Bundle with at least one entry
  resource_type=$(echo "$search_body" | grep -o '"resourceType"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | grep -o '"[^"]*"$' | tr -d '"')
  total=$(echo "$search_body" | grep -o '"total"[[:space:]]*:[[:space:]]*[0-9]*' | head -1 | grep -o '[0-9]*$')

  if [ "$resource_type" = "Bundle" ] && [ "${total:-0}" -gt 0 ]; then
    pass "I3 - FHIR search returned Bundle with ${total} result(s) (hapi schema and hfj_resource table confirmed)"
  else
    fail "I3 - FHIR search returned unexpected response (resourceType: '${resource_type}', total: '${total:-0}'). Expected Bundle with total > 0"
  fi
else
  fail "I3 - GET /fhir/Patient?family=${PATIENT_FAMILY} returned HTTP ${search_http_code} (expected 200)"
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
