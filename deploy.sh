#!/bin/sh
# Push code + any keys in .env to the Azure web app (free F1 plan).
set -e
cd "$(dirname "$0")"
. ./.azure
# Copy non-empty keys from .env into Azure app settings (never into the zip)
SETTINGS=$(grep -E '^(ANTHROPIC_API_KEY|ANTHROPIC_WORKSPACE_ID|SCHWAB_APP_KEY|SCHWAB_APP_SECRET)=.+' .env || true)
[ -n "$SETTINGS" ] && az webapp config appsettings set -g options-radar-rg -n "$AZURE_APP" --settings $SETTINGS -o none
# Dashboard password lives in .azure
[ -n "$DASHBOARD_PASSWORD" ] && az webapp config appsettings set -g options-radar-rg -n "$AZURE_APP" --settings "DASHBOARD_PASSWORD=$DASHBOARD_PASSWORD" -o none
ZIP=$(mktemp -d)/radar.zip
zip -qr "$ZIP" server.py requirements.txt radar static vendor/Kronos/model vendor/Kronos/LICENSE -x "*/__pycache__/*"
az webapp deploy -g options-radar-rg -n "$AZURE_APP" --src-path "$ZIP" --type zip -o none
echo "Deployed: https://$AZURE_APP.azurewebsites.net"
