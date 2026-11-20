# GitHub publishing steps

1. Review `PUBLICATION_CHECKLIST.md`, particularly the historical notebook, distribution rights and license. Do not publish until those are resolved.
2. Create a new empty GitHub repository (do not auto-add a README or license).
3. Extract this archive, open a terminal in the extracted folder and run:

```bash
git init
git add .
git commit -m "Release Q8 checkpoint reproducibility materials"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

4. Check GitHub's file-size limits and inspect uploaded content. The supplied archive contains the preserved V5 inputs, not the reserved test split.
5. Add author details, paper citation, chosen license and Zenodo DOI after they exist. Archive a fixed GitHub release on Zenodo if desired.

The package is ready for a **pre-publication review**, not an assertion that publication or archiving has already occurred.
