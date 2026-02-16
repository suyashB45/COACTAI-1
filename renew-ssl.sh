#!/bin/bash

LOG_FILE="/var/log/certbot-renew.log"
PROJECT_DIR="/CoAct.AI"
DOCKER="/usr/bin/docker"

echo "$(date): Starting certificate renewal check" >> $LOG_FILE

# Renew certificates
if certbot renew --quiet; then
    echo "$(date): Certbot renew command executed" >> $LOG_FILE

    # Reload nginx inside container (no TTY)
    cd $PROJECT_DIR || exit
    $DOCKER compose exec -T frontend nginx -s reload

    echo "$(date): Nginx reloaded successfully" >> $LOG_FILE
else
    echo "$(date): Certbot renewal failed" >> $LOG_FILE
fi
echo "$(date): Certificate renewal check completed" >> $LOG_FILE