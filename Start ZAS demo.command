#!/bin/zsh
cd -- "${0:A:h}"
python3 scripts/start_demo.py
if [ $? -ne 0 ]; then
  read -r "?Druk op Enter om te sluiten."
fi
