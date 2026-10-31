# Een andere LLM-API aansluiten

De app ondersteunt drie API-routes. Voor een andere aanbieder hoef je de
annotatiecode niet te veranderen: zet provider, endpoint, model en sleutel in
`backend/.env` en herstart de container met `docker compose up -d --force-recreate app`.
Een gewone `restart` leest gewijzigde Docker-omgevingsvariabelen niet opnieuw in.

| Route | API-formaat | Configuratievoorbeeld |
| --- | --- | --- |
| OpenAI | Responses, eventueel Chat Completions | [`backend/.env.example`](../backend/.env.example) |
| OpenAI-compatibel | Chat Completions, eventueel Responses | [`backend/.env.openai-compatible.example`](../backend/.env.openai-compatible.example) |
| Anthropic | Messages met een verplichte tool voor het JSON-resultaat | [`backend/.env.anthropic.example`](../backend/.env.anthropic.example) |

Een willekeurige andere REST-API is niet automatisch compatibel: die heeft een
adapter of compatibele gateway nodig. De adapter ondersteunt het protocol; de
kwaliteit van Nederlandse medische extractie en codekeuze moet je per model
beoordelen. Niet ieder model ondersteunt ieder formaat voor gestructureerde uitvoer.

## OpenAI

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
LLM_API_KEY=your-key
LLM_JSON_MODE=json_schema
LLM_REASONING_EFFORT=low
```

`OPENAI_API_KEY` wordt als alternatief gelezen als `LLM_API_KEY` leeg is.
De standaardroute is Responses. Kies `LLM_API=chat_completions` als je dat
API-formaat nodig hebt. Laat `LLM_REASONING_EFFORT` leeg als het model deze
parameter niet ondersteunt.

## Een compatibel endpoint, bijvoorbeeld lokaal Ollama

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=local
LLM_MODEL=your-installed-model
LLM_API=chat_completions
LLM_JSON_MODE=json_schema
LLM_REASONING_EFFORT=
DEFAULT_EMBEDDER=biolord
```

Vervang de modelnaam door een daadwerkelijk geïnstalleerd model. Bij een native
installatie gebruik je `http://127.0.0.1:11434/v1`. Vanuit Docker verwijst
`localhost` naar de container zelf; `host.docker.internal` verwijst naar de host.
Compose bevat hiervoor ook de Linux-hostmapping. De LLM-server moet voor de
container bereikbaar zijn; de juiste netwerkbinding en firewall blijven
onderdeel van de interne inrichting. Een beheerde API of eigen gateway kan
uiteraard een ander adres en een echte sleutel gebruiken.

De app installeert de lokale LLM-server of diens model niet. Na het downloaden
van BioLORD en het lokale taalmodel kan deze combinatie zonder externe LLM-API
werken. Er wordt nooit automatisch uitgeweken naar OpenAI. De OpenAI-sleutel
wordt niet als fallback naar een compatibele aanbieder verstuurd.

## Anthropic Messages

```dotenv
LLM_PROVIDER=anthropic
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_API_KEY=your-key
LLM_MODEL=your-claude-model
LLM_API=messages
LLM_JSON_MODE=json_schema
LLM_REASONING_EFFORT=
```

Gebruik een model dat in je eigen account beschikbaar is. `ANTHROPIC_API_KEY`
wordt als alternatief gelezen als `LLM_API_KEY` leeg is. Deze route gebruikt een
verplichte tool met het gevraagde schema als gestructureerd antwoord; er wordt
geen externe tool uitgevoerd. OpenAI-specifieke reasoning-parameters worden
niet naar Anthropic gestuurd.

## Als het endpoint geen strikt JSON Schema ondersteunt

Gebruik bij OpenAI-compatibele endpoints bij voorkeur `LLM_JSON_MODE=json_schema`.
Als het model alleen JSON-modus ondersteunt, kies `json_object`. Met `prompt`
wordt geen `response_format`-parameter meegestuurd: het schema staat dan in de
instructie. Die laatste twee opties zijn afhankelijker van modelgedrag.

Bij alle routes valideert de app het ontvangen JSON tegen het schema. Ongeldige,
afgebroken of geweigerde antwoorden leveren een fout op. Een gekozen concept
moet bovendien voorkomen in de opgehaalde kandidatenlijst. Er is geen stilzwijgende
fallback naar een andere provider of een minder strikte uitvoermodus.

`LLM_MAX_TOKENS` begrenst het antwoord (standaard 4096).
`LLM_TIMEOUT_SECONDS` begrenst het wachten per aanvraag (standaard 120 seconden).
Voor een trager lokaal model kun je de timeout verhogen en `SELECT_WORKERS`
verlagen om gelijktijdige aanvragen te beperken.

## De koppeling testen

```bash
docker compose run --rm --no-deps app python -m app.check --config-only --probe-llm
```

Dit verstuurt één synthetische JSON-test naar de ingestelde provider. Er worden
geen patiëntgegevens of voorbeeldverslagen verstuurd. Deze test controleert de
verbinding en het uitvoerformaat, niet de medische kwaliteit. `/api/readiness`
controleert de lokale configuratie en aanwezigheid van de zoekindex, zonder
externe API-aanroep. De extractie- en resultaatcaches worden gescheiden per
provider, endpoint, model en uitvoerinstellingen.

## LLM en embeddings zijn gescheiden

BioLORD is de standaard voor lokale zoekopdrachten. Bij een andere LLM-aanbieder
heb je daarvoor geen OpenAI-sleutel nodig. Kies je daarnaast de optionele
OpenAI-zoekindex, dan blijven die embeddings via OpenAI lopen met een aparte
`OPENAI_API_KEY`. `LLM_BASE_URL` en `LLM_API_KEY` veranderen die embeddingroute niet.

API-referenties: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Ollama OpenAI-compatibiliteit](https://docs.ollama.com/api/openai-compatibility) en
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create).
