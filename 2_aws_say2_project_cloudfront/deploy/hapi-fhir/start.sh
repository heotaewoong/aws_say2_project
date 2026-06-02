#!/bin/bash
set -euo pipefail

# ============================================================
# HAPI FHIR Start Script
# Runs on the EC2 HOST (not inside the container).
# Retrieves database password from AWS Secrets Manager and
# passes it to docker compose as an environment variable.
# ============================================================

AWS_REGION="${AWS_REGION:-ap-northeast-2}"
SECRET_NAME="${SECRET_NAME:-rare-link-ai/aurora/hapi-user}"

echo "Retrieving database credentials from Secrets Manager..."
echo "  Secret: ${SECRET_NAME}"
echo "  Region: ${AWS_REGION}"

# Fetch the secret value from AWS Secrets Manager
SECRET_VALUE=$(aws secretsmanager get-secret-value \
    --secret-id "${SECRET_NAME}" \
    --region "${AWS_REGION}" \
    --query 'SecretString' \
    --output text 2>/tmp/secrets-error.log) || {
    echo "ERROR: Failed to retrieve secret '${SECRET_NAME}' from Secrets Manager (region: ${AWS_REGION})" >&2
    if [ -f /tmp/secrets-error.log ]; then
        cat /tmp/secrets-error.log >&2
    fi
    exit 1
}

if [ -z "${SECRET_VALUE}" ]; then
    echo "ERROR: Secret '${SECRET_NAME}' returned an empty value" >&2
    exit 1
fi

echo "Successfully retrieved database credentials."

# Export and run docker compose
export SPRING_DATASOURCE_PASSWORD="${SECRET_VALUE}"
docker compose up -d

echo ""
echo "HAPI FHIR container started. Check status with:"
echo "  docker compose logs -f hapi-fhir"
echo "  docker inspect --format='{{.State.Health.Status}}' hapi-fhir"
