# SNOMED CT Annotator

**Van klinische tekst naar aanklikbare SNOMED CT-codes.** Dit ZAS-prototype helpt
zorgverleners en terminologieteams om vrije Nederlandstalige tekst te structureren
en voorgestelde codes te controleren.

- **Annoteren:** medische begrippen worden gemarkeerd met hun SNOMED CT-code.
- **Controleren:** klik op een begrip, bekijk alternatieven en pas de code aan.
- **Zoeken:** zoek rechtstreeks in SNOMED CT, ook met Nederlandse termen.
- **Inzicht:** bekijk de verwerkingsstappen en gebruikte AI-instructies.

### Geannoteerde tekst

Fictief voorbeeld: de oorspronkelijke tekst blijft leesbaar; rechts staan de
voorgestelde code en alternatieven.

![Geannoteerde fictieve tekst met aanklikbare SNOMED CT-codes en alternatieven](docs/images/annotated-text.jpg)

### Hoe werkt het?

AI herkent medische begrippen, een zoekindex vindt passende SNOMED CT-concepten
en een optionele AI-stap kiest de beste kandidaat in de context. De gebruiker
kan die keuze controleren en aanpassen.

![Pipeline van klinische tekst tot controleerbare SNOMED CT-annotaties](docs/images/pipeline.jpg)

### Binnen de zorginstelling installeren

De IT-dienst kan de webapp op één interne computer of server draaien met Docker.
Een GPU of aparte terminologieserver is niet nodig. Nodig zijn een geldige
**SNOMED CT International RF2-release** en een **OpenAI API-sleutel** met
internettoegang voor nieuwe annotaties.

```bash
git clone https://github.com/stighellemans/snomed-annotator.git
cd snomed-annotator
cp backend/.env.example backend/.env
# Vul OPENAI_API_KEY in en plaats de RF2-zip in snomed/.
docker compose run --build --rm index  # eenmalig de lokale zoekindex opbouwen
docker compose up -d --build app
```

Open daarna **http://localhost:5180**. De index blijft bewaard bij herstarten.
[Installatie, intern delen en configuratie →](docs/installation.md)

**Voor een pilot:** de webapp en BioLORD-zoekindex draaien lokaal, maar nieuwe
annotaties sturen tekst naar OpenAI, ook wanneer de optionele AI-keuze uitstaat.
Het prototype is niet klinisch gevalideerd. Handmatige correcties blijven alleen
in het huidige scherm; er is nog geen EPD-koppeling of gebruikersbeheer.

Deze repo bevat nu de ZAS-webapp; de eerdere versie met scripts blijft
[beschikbaar in de Git-historiek](https://github.com/stighellemans/snomed-annotator/tree/65bc89466491a0a521526eada0b07dd109eb2a2d).
