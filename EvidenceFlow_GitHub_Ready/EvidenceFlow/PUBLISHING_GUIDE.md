# Publishing Guide: Pushing EvidenceFlow to GitHub

## Option 1: Upload via GitHub Web Interface (Fastest)
1. Extract `EvidenceFlow_GitHub_Ready.zip` on your target machine.
2. Open the extracted `EvidenceFlow` folder so you see `src`, `tests`, `README.md`, etc.
3. Go to your empty GitHub repository (`Agent`) in your browser and click **uploading an existing file**.
4. Drag and drop all files and folders from inside `EvidenceFlow` into the upload area.
5. Add commit message `Initial commit: EvidenceFlow Agentic AI Assistant` and click **Commit changes**.

## Option 2: Push via Git CLI
```bash
cd EvidenceFlow
git init
git add .
git commit -m "Initial commit: EvidenceFlow Agentic AI Assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/Agent.git
git push -u origin main
```
