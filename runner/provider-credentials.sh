#!/bin/bash
# Sourced by the runner: credentials stay in the setup shell and tmpfs vault.
clear_provider_environment() {
  local pair key_env
  local -a pairs
  IFS=, read -ra pairs <<<"${BIALY_HOSTED_PROVIDERS:-}"
  for pair in "${pairs[@]}"; do
    [ -n "$pair" ] || continue
    key_env=${pair#*:}
    [[ "$key_env" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || return 1
    unset "$key_env"
  done
  unset GITHUB_SEARCH_API_KEY GITLAB_SEARCH_API_KEY STACKEXCHANGE_API_KEY
}

store_provider_credential() {
  local provider=$1 key_env=$2 token_file=$3 provider_path=${4:-providers}
  [[ "$key_env" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || return 1
  [ -n "${!key_env:-}" ] || return 1
  # Neither the provider key nor the bearer token appears in a process argument.
  printf '%s' "${!key_env}" | jq -Rs '{api_key: .}' |
    curl -s -o /dev/null -w '%{http_code}' -X PUT \
      -H "@$token_file" -H 'Content-Type: application/json' --data-binary @- \
      "http://127.0.0.1:8850/v1/$provider_path/$provider/credential"
}
