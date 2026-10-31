# Installatie en beheer

## Docker: één webapp en één lokale zoekindex

Installeer Docker met Compose. Gebruik een geldige **SNOMED CT International
RF2-release**: de importer verwacht de Engelstalige International Snapshot-bestanden.
Een Belgische extensie zonder die bestanden is niet voldoende. De release en
zoekindex worden niet meegeleverd.

Voer vanuit de hoofdmap uit:

```bash
cp backend/.env.example backend/.env
```

Vul `OPENAI_API_KEY` in `backend/.env` in. Plaats precies één RF2-release-zip in
`snomed/` en bouw de index:

```bash
docker compose run --build --rm index
docker compose up -d --build app
```

De standaardinstallatie gebruikt **BioLORD** voor lokale zoekopdrachten en
**OpenAI** voor herkenning van begrippen en de optionele codekeuze. Het bouwen
van deze BioLORD-index gebruikt geen betaalde OpenAI-embeddings. De eerste build
downloadt het model en verwerkt de volledige release; dat kan lang duren en
vraagt meer geheugen dan een gewone zoekopdracht. Een GPU is niet nodig.

De app is bereikbaar op <http://localhost:5180>. `backend`, `frontend`, de
RF2-importer en beide indexbouwers staan in deze repo. Andere projecten of
SNOMED-servers zijn niet nodig.

```bash
docker compose ps
docker compose logs -f app
docker compose stop app       # stoppen, data behouden
docker compose up -d app      # opnieuw starten
```

De volumes `annotation-data` en `model-cache` bewaren de voorbereide terminologie,
index, lokale resultaatcaches en het model. `docker compose down` behoudt deze
volumes; `docker compose down -v` verwijdert ze. De caches bevatten ingevoerde
tekst: neem ze mee in het interne toegangs- en bewaarbeleid. De drie meegeleverde
voorbeelden zijn fictief; er worden geen eerder berekende annotaties meegeleverd.

## Delen binnen de zorginstelling

Docker bindt standaard alleen aan `127.0.0.1`. Laat de IT-dienst de app voor
collega's ontsluiten via de bestaande interne reverse proxy, met HTTPS en de
gebruikelijke aanmelding. De app zelf heeft geen authenticatie, rollen of
auditlog. Gebruik één applicatieproces, want de eenvoudige JSON-caches zijn niet
geschikt voor meerdere processen die dezelfde bestanden schrijven.

Nieuwe annotaties sturen de volledige tekst naar OpenAI voor extractie. Bij
ingeschakelde **LLM arbitrator** worden tekst en kandidaten ook gebruikt voor de
codekeuze. **Top-1 vector** slaat alleen die tweede AI-stap over. Rechtstreeks
zoeken met BioLORD werkt lokaal nadat het model is gedownload. De optionele
OpenAI-zoekmethode verstuurt ook de zoekterm. Zorg dat deze gegevensstroom past
binnen de afspraken van de instelling voordat je patiëntgegevens gebruikt.

Negaties worden via de AI-instructie overgeslagen; dat is geen gevalideerde
garantie. De getoonde scores zijn geen bewezen klinische betrouwbaarheden.
Correcties in de annotatieweergave worden nog niet opgeslagen of geëxporteerd.

## Optioneel: OpenAI-zoekindex

Voor de standaardinstallatie volstaat BioLORD. Om ook de OpenAI-optie in de
interface beschikbaar te maken:

```bash
docker compose run --rm index python -m indexing.build --embedders openai
docker compose restart app
```

Hierbij gaat het om SNOMED-termen uit de release. Deze indexopbouw maakt betaalde
API-aanroepen voor de volledige terminologie. Test desgewenst eerst met
`--limit 2000` in een aparte testinstallatie; zo'n beperkte index is onvolledig.
Verwijder een testindex vóór je een volledige index bouwt.

## Zonder Docker: lokaal ontwikkelen

Gebruik Python 3.12 en Node.js 22 of nieuwer. Vanuit de hoofdmap:

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
# Vul de sleutel in en plaats de RF2-zip in snomed/.
cd backend
.venv/bin/python -m indexing.build --embedders biolord
cd ../frontend
npm ci
npm run build
cd ..
python3 scripts/start_demo.py
```

De launcher opent <http://127.0.0.1:5180> en start de app op de achtergrond.
Op macOS kun je na de installatie ook dubbelklikken op `Start ZAS demo.command`.
De log staat in `.runtime/demo.log` en de PID in `.runtime/demo.pid`.
Op Windows kun je na installatie vanuit `backend` rechtstreeks starten met
`.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 5180`.

Op Apple Silicon: zet `EMBEDDER_DEVICE=mps` in `.env` vóór de indexopbouw en het
starten. Bij ontwikkeling kun je de backend op poort 8000 draaien en in
`frontend` `npm run dev` gebruiken; Vite stuurt `/api` door naar de backend.

## Configuratie

Beide indexbouwers en de app lezen `backend/.env`. Standaardpaden zijn absoluut
afgeleid van de projectmap; een aparte configuratie voor andere projecten is
niet nodig.

| Variabele | Standaard / gebruik |
| --- | --- |
| `OPENAI_API_KEY` | Vereist voor nieuwe annotaties en OpenAI-zoeken |
| `LLM_MODEL` | `gpt-5.4-mini`; kies een model met Responses en structured outputs |
| `LLM_REASONING_EFFORT` | `low`; leeg laten als het gekozen model dit niet ondersteunt |
| `EMBEDDER_MODEL` | `FremyCompany/BioLORD-2023-M` |
| `EMBEDDER_DEVICE` | `cpu`; lokaal ook `cuda` of `mps` |
| `DEFAULT_EMBEDDER` | `biolord` |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` |
| `TOP_K` | `10` kandidaten per begrip |
| `MAX_SPANS` | `40` begrippen per tekst |
| `MIN_CONFIDENCE` | `0.4`; technische drempel, geen gevalideerde kans |
| `SELECT_WORKERS` | `8` gelijktijdige AI-aanroepen voor codekeuze |
| `INDEX_DIR` | `backend/data/indexes/biolord/` |
| `OPENAI_INDEX_PATH` | `backend/data/indexes/openai/faiss.index` |
| `OPENAI_META_PATH` | `backend/data/indexes/openai/store.meta.sqlite` |

Het embeddingmodel voor zoeken moet overeenkomen met dat van de index. Bouw bij
een model- of releasewijziging de index opnieuw op en wis de lokale
resultaatcaches; die herkennen een gewijzigde terminologierelease niet vanzelf.
Voor een nieuwe RF2-release: vervang de zip en gebruik
`python -m indexing.build --embedders biolord --force-prepare` via de indexservice.
Bij een andere LLM-instelling: herstart de app en wis de caches als je nieuwe
resultaten wilt, want niet alle instellingen zitten in de cachesleutel.

## Overgang van het oudere prototype

De GitHub-versie vóór deze update was een CLI voor vectorgebaseerde koppeling en
analysebestanden. De ZAS-versie voegt een React-interface, FastAPI-service,
BioLORD, aanklikbare annotaties, handmatige codekeuze en een pipelineweergave toe.
De zelfstandig werkende RF2-importer en indexbouwers zijn mee overgenomen.
De oude CLI en broncode blijven in commit
[`65bc894`](https://github.com/stighellemans/snomed-annotator/tree/65bc89466491a0a521526eada0b07dd109eb2a2d).
Oude CLI-commando's en vectorbestanden zijn geen onderdeel van de nieuwe installatie.

De oorspronkelijke ZAS-demo op deze Mac gebruikte externe indexpaden en een
gedeelde Python-omgeving. Deze publieke versie gebruikt eigen indexpaden en een
eigen installatie; die lokale demo blijft afzonderlijk bestaan.

De broncode valt onder de [MIT-licentie](../LICENSE). De SNOMED CT-release en
modelbestanden vallen onder hun eigen licenties en worden niet gepubliceerd.

Het installatiepad is getest met Docker op Linux ARM64, een kleine RF2-testset,
lokale BioLORD-zoekopdrachten en beide annotatiemodi met fictieve tekst. De
screenshots gebruiken de bestaande volledige BioLORD-index. Een nieuwe volledige
RF2-indexopbouw en koppeling met de infrastructuur van een zorginstelling zijn
niet opnieuw uitgevoerd.
