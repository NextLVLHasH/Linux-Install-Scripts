#!/usr/bin/env bash
# One-shot mTLS cert generator for the HasH AI stack. Produces:
#   ca.crt      — root CA. Trusted by both the pod's Caddy and the API client.
#   server.crt  — pod-side TLS cert (CN = $SERVER_CN). Caddy presents this on incoming TLS.
#   server.key  — pod-side private key.
#   client.pfx  — API-side PKCS#12 bundle (client cert + key, password-protected). The C#
#                 HttpClient loads this and presents it during the TLS handshake.
#
# Run this on a TRUSTED box (your laptop). Copy:
#   ca.crt + server.crt + server.key  →  pod  /etc/hashai/
#   ca.crt + client.pfx               →  API  /etc/hashai/  (or anywhere the C# can read)
# Then DELETE ca.key from disk after issuing both certs — the only thing that needs to
# survive is the public ca.crt.
#
# Usage:
#   SERVER_CN=api.example.com  ./gen-mtls-certs.sh           # public DNS name
#   SERVER_CN=10.20.30.40      ./gen-mtls-certs.sh           # Runpod TCP-proxy host
#   PFX_PASSWORD=$(openssl rand -hex 16) ./gen-mtls-certs.sh # explicit pfx password
#
# Required: openssl (>= 3.0).

set -euo pipefail

OUT_DIR="${OUT_DIR:-./mtls-out}"
SERVER_CN="${SERVER_CN:?SERVER_CN env var required (the hostname/IP the API will dial)}"
CLIENT_CN="${CLIENT_CN:-hashai-api-client}"
DAYS_CA="${DAYS_CA:-3650}"           # 10y CA
DAYS_LEAF="${DAYS_LEAF:-825}"        # ~2y for leaf certs (browsers cap at 825)
PFX_PASSWORD="${PFX_PASSWORD:-$(openssl rand -hex 16)}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "==> Generating root CA (HasH AI mTLS)"
openssl genrsa -out ca.key 4096 2>/dev/null
openssl req -x509 -new -nodes -key ca.key -sha256 -days "$DAYS_CA" \
    -subj "/CN=HasH AI mTLS Root" -out ca.crt

echo "==> Generating server cert (CN=$SERVER_CN)"
openssl genrsa -out server.key 2048 2>/dev/null
cat > server.cnf <<EOF
[req]
distinguished_name = dn
req_extensions = v3
prompt = no

[dn]
CN = $SERVER_CN

[v3]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @san

[san]
DNS.1 = $SERVER_CN
EOF
# Add IP SAN if SERVER_CN parses as an IP.
if [[ "$SERVER_CN" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "IP.1 = $SERVER_CN" >> server.cnf
fi
openssl req -new -key server.key -config server.cnf -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -days "$DAYS_LEAF" -sha256 -extfile server.cnf -extensions v3 -out server.crt
rm -f server.csr server.cnf ca.srl

echo "==> Generating client cert (CN=$CLIENT_CN)"
openssl genrsa -out client.key 2048 2>/dev/null
cat > client.cnf <<EOF
[req]
distinguished_name = dn
req_extensions = v3
prompt = no

[dn]
CN = $CLIENT_CN

[v3]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
EOF
openssl req -new -key client.key -config client.cnf -out client.csr
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -days "$DAYS_LEAF" -sha256 -extfile client.cnf -extensions v3 -out client.crt
rm -f client.csr client.cnf ca.srl

echo "==> Bundling client.pfx"
openssl pkcs12 -export -out client.pfx \
    -inkey client.key -in client.crt -certfile ca.crt \
    -passout "pass:${PFX_PASSWORD}"

# Lock down private files to owner-only — they are the keys to the kingdom.
chmod 600 ca.key server.key client.key client.pfx 2>/dev/null || true

echo
echo "──────────────────────────────────────────────────────────"
echo " Files in $OUT_DIR/"
echo
echo " Pod (Caddy):"
echo "   /etc/hashai/ca.crt       (← ca.crt)"
echo "   /etc/hashai/server.crt   (← server.crt)"
echo "   /etc/hashai/server.key   (← server.key)"
echo
echo " API (HasHAI-API):"
echo "   client.pfx"
echo "   PFX password (set as the env var named in ClientCertPasswordEnvVar):"
echo "     $PFX_PASSWORD"
echo
echo " KEEP SAFE / OFFLINE:"
echo "   ca.key   (sign future certs with this — never copy it to a server)"
echo "──────────────────────────────────────────────────────────"
