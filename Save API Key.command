#!/bin/zsh
# Saves the Anthropic API key that is currently on the clipboard into .env.
# Usage: copy your key (starts with sk-ant-) in the browser, then double-click this file.
cd "$(dirname "$0")"
key="$(pbpaste | tr -d '[:space:]')"
if [[ "$key" != sk-ant-* ]]; then
  echo ""
  echo "The clipboard doesn't contain an Anthropic key (it should start with sk-ant-)."
  echo "Copy the key from the Claude Console first, then double-click this file again."
  echo ""
  read -k1 "?Press any key to close."
  exit 1
fi
print -r -- "ANTHROPIC_API_KEY=$key" > .env
chmod 600 .env
unset key
pbcopy < /dev/null   # clear the clipboard so the key isn't pasted somewhere by accident
echo ""
echo "Saved. Your key is in $(pwd)/.env (only you can read it)."
echo "You can close this window."
echo ""
