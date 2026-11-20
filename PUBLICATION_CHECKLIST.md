# Before publishing

- [ ] Review the recovered historical notebook for secrets, personal information, external data, and irrelevant exploratory material. Remove it from the public distribution if unsuitable; retain its hash/provenance in an appropriate record.
- [ ] Verify you have the rights to distribute the checkpoint, data, and source code.
- [ ] Choose a software license and (if appropriate) a separate data license.
- [ ] Run `python verify_source_manifest.py` and all five verification scripts in a clean environment.
- [ ] Confirm generated reports agree with the paper's numerical tables and figures.
- [x] Add author information and preliminary citation metadata (`README.md`, `CITATION.cff`).
- [ ] Add repository URL, final paper citation and Zenodo DOI **only once available**.
- [ ] Keep the reserved test split unopened until the confirmatory evaluation protocol is fixed.
- [ ] State explicitly that the repository does not reproduce training from initialization.
- [ ] Optionally add figure-generation scripts after verifying they produce the manuscript figures exactly; they are not established by this V5 package.
