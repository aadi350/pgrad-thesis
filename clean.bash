#/usr/bin/zsh

find . -type d -name __pycache__  -exec rm -rf {} \;
find . -type d -name .ipynb_checkpoints  -exec rm -rf {} \;
find . -type d -name .vscode -exec rm -rf {} \;
