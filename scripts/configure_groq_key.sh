#!/usr/bin/env bash
set -euo pipefail

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/physical-ai-runtime"
config_file="$config_dir/groq.env"

mkdir -p "$config_dir"
chmod 700 "$config_dir"

read -r -s -p "Groq API key: " groq_api_key
printf '\n'

if [[ -z "$groq_api_key" ]]; then
  echo "No key provided; configuration unchanged." >&2
  exit 1
fi

umask 077
printf 'GROQ_API_KEY=%s\n' "$groq_api_key" > "$config_file"
chmod 600 "$config_file"

echo "Saved local Groq configuration: $config_file"
