# Drop your SNOMED CT release here

Place a SNOMED CT **International RF2** release zip in this folder, e.g.:

```
snomed/SnomedCT_InternationalRF2_PRODUCTION_20260201T120000Z.zip
```

Use a release you are entitled to use. Release files are not committed; this
folder ignores every file except this README.

Then build the indices from the repo root:

```bash
docker compose run --build --rm index  # local BioLORD index; no OpenAI embedding calls
```

Keep exactly one zip here. The builder parses it once into `data/prepared/`,
then writes `data/indexes/biolord/` in the backend data volume. The optional
OpenAI index and native installation are described in
[the installation guide](../docs/installation.md).
