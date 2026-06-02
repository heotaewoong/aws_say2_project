#!/bin/bash
set -euo pipefail

# ============================================================
# HAPI FHIR Entrypoint Script
# Retrieves database password from AWS Secrets Manager and
# starts the HAPI FHIR JPA Server.
# ============================================================

AWS_REGION="${AWS_REGION:-ap-northeast-2}"
SECRET_NAME="${SECRET_NAME:-rare-link-ai/aurora/hapi-user}"

echo "Retrieving database credentials from Secrets Manager..."
echo "  Secret: ${SECRET_NAME}"
echo "  Region: ${AWS_REGION}"

# Fetch the secret value from AWS Secrets Manager
# The secret is a plain string (the password), not a JSON object.
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

# Validate that we got a non-empty value
if [ -z "${SECRET_VALUE}" ]; then
    echo "ERROR: Secret '${SECRET_NAME}' returned an empty value" >&2
    exit 1
fi

export SPRING_DATASOURCE_PASSWORD="${SECRET_VALUE}"
echo "Successfully retrieved database credentials."

# Start HAPI FHIR JPA Server
# Using exec to replace the shell process for proper signal handling
exec java -jar /app/main.war
