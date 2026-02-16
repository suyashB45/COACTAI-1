#!/bin/bash

# init-letsencrypt.sh
# Automated SSL Certificate Setup for CoAct.AI

domains=(coact-ai.com www.coact-ai.com)
email="coactai@outlook.com"
data_path="./certbot"
rsa_key_size=4096

echo "### Starting Let's Encrypt Setup for ${domains[*]} ###"

# 1. Clean up old certificates to avoid conflicts and ensure a fresh start
echo "Cleaning up old configuration..."
rm -rf "$data_path/conf/live/${domains[0]}"
rm -rf "$data_path/conf/archive/${domains[0]}"
rm -rf "$data_path/conf/renewal/${domains[0]}.conf"

# 2. Create required directories
echo "Creating directories..."
mkdir -p "$data_path/conf/live/${domains[0]}"
mkdir -p "$data_path/www"

# 3. Download TLS parameters (security best practices)
if [ ! -f "$data_path/conf/options-ssl-nginx.conf" ]; then
    echo "Downloading TLS parameters..."
    curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$data_path/conf/options-ssl-nginx.conf"
    curl -s https://raw.githubusercontent.com/certbot/certbot/master/certbot/certbot/ssl-dhparams.pem > "$data_path/conf/ssl-dhparams.pem"
fi

# 4. Create dummy certificate (required for Nginx to start the first time)
echo "Generating dummy certificate..."
openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1 \
  -keyout "$data_path/conf/live/${domains[0]}/privkey.pem" \
  -out "$data_path/conf/live/${domains[0]}/fullchain.pem" \
  -subj "/CN=localhost" 2>/dev/null

# 5. Start Nginx
echo "Starting Nginx..."
docker compose up -d frontend

echo "Waiting for Nginx to launch (10s)..."
sleep 10

# 6. Remove dummy certificate (so Certbot can write the real one)
echo "Removing dummy certificate..."
rm "$data_path/conf/live/${domains[0]}/privkey.pem"
rm "$data_path/conf/live/${domains[0]}/fullchain.pem"

# 7. Request real certificate
echo "Requesting real certificates from Let's Encrypt..."
docker compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    --email $email \
    -d ${domains[0]} -d ${domains[1]} \
    --rsa-key-size $rsa_key_size \
    --agree-tos \
    --force-renewal \
    --no-eff-email" certbot

# 8. Reload Nginx to apply the new certificate
echo "Reloading Nginx..."
docker compose exec frontend nginx -s reload

echo "### Setup Complete! ###"
echo "Your site should now be available at https://${domains[0]}"
