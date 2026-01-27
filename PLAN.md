# Plan d'amelioration et migration V2

Objectif
- Rendre la structure plus claire (code vs data).
- Centraliser la config et les chemins.
- Reduire les duplications et les risques d'erreur.
- Garder le projet fonctionnel a chaque etape.

Perimetre
- Ads (Meta): extraction, exports Excel, Google Sheets, dashboard.
- Organic FB/IG: extraction, exports Excel, Google Sheets, dashboard.
- Auth/tokens et config globale.

Principes
- Migration progressive, pas de big-bang.
- Chaque etape doit se terminer par des checks concrets avant de continuer.
- Aucun changement destructif sur les donnees sans confirmation.

Structure cible (resume)
- src/core: config, paths, logging, http session
- src/meta/ads: fetch, export, sheets
- src/meta/organic: fb_fetch, ig_fetch, export, sheets
- src/dashboards: streamlit apps
- src/ai: agent
- data/: json/excel/stories
- scripts/: points d'entree

Etat actuel (apres actions)
- Arborescence V2 creee (src/, data/, scripts/)
- Modules Ads/Organic V2 crees et relies aux chemins data/
- Dashboards legacy mis a jour pour lire data/ + agent IA V2
- Versions Graph API centralisees dans core/config.py
- Retry HTTP utilise dans Ads/Organic
- Nettoyage legacy effectue (artefacts + anciens pipelines)

Checklist globale des ameliorations
- [x] Centraliser les variables d'environnement et versions Graph API
- [x] Normaliser les chemins (core/paths.py)
- [x] Separation code vs data (dashboards lisent data/)
- [x] Regrouper la logique retry HTTP (core/http.py)
- [x] Ajouter validations de fichiers avant etapes suivantes
- [x] Mettre a jour README + commandes
- [x] Ajouter .gitignore pour artefacts et secrets
- [x] Supprimer les anciens pipelines legacy (campaigns/*, organic_performance/*)
- [ ] Verifier absence de secrets en dur encore utiles (auths legacy)

Processus par etapes (avec checks)

Etape 1 - Cartographie (etat actuel)
Actions
- Lister points d'entree actuels (scripts principaux, dashboards)
- Lister variables d'env utilisees par domaine
- Lister fichiers JSON/Excel produits et consommations
Checks pour valider
- [x] Liste des scripts et dashboards validee
- [x] Liste des variables d'env validee
- [x] Liste des fichiers de data validee

Etape 2 - Preparations
Actions
- Creer arborescence cible (src/, data/, scripts/)
- Ajouter core/paths.py et core/config.py (structure vide + conventions)
- Definir schema de chemins data/* (ads, organic fb, organic ig, stories)
Checks pour valider
- [x] Arborescence creee
- [x] Conventions de chemins ecrites
- [x] Plan de mapping chemins existants -> nouveaux chemins OK

Etape 3 - Migration Ads (progressive)
Actions
- Deplacer campaigns/main_campaigns_paid.py -> src/meta/ads/fetch.py
- Deplacer campaigns/json_to_excel.py -> src/meta/ads/export.py
- Deplacer campaigns/google_sheet_automatisation.py -> src/meta/ads/sheets.py
- Adapter imports + chemins via core/paths.py
- Ajouter script scripts/run_ads.py
Checks pour valider
- [ ] Extraction Ads fonctionne (JSON generes)
- [ ] Export Excel fonctionne
- [ ] Envoi Google Sheets fonctionne

Etape 4 - Migration Organic FB/IG
Actions
- Deplacer organic_performance/connect_api_insta_by_meta.py -> src/meta/organic/common.py
- Deplacer main_facebook_post_organic.py -> src/meta/organic/fb_fetch.py
- Deplacer main_instagram_organic.py -> src/meta/organic/ig_fetch.py
- Deplacer format_json_to_excel.py -> src/meta/organic/export.py
- Deplacer google_sheet_automatisation.py -> src/meta/organic/sheets.py
- Adapter imports + chemins via core/paths.py
- Ajouter scripts/run_organic_fb.py et run_organic_ig.py
Checks pour valider
- [ ] Extraction FB fonctionne (JSON generes)
- [ ] Extraction IG fonctionne (JSON generes)
- [ ] Export Excel fonctionne
- [ ] Envoi Google Sheets fonctionne

Etape 5 - Dashboards + AI agent
Actions
- Deplacer dashboards streamlit dans src/dashboards/ (wrappers)
- Adapter imports (ai/agent, paths)
- Verifier chargement des JSON/Excel
Checks pour valider
- [ ] Dashboard Ads s'ouvre sans erreur
- [ ] Dashboard IG fonctionne
- [ ] (Optionnel) Agent IA se charge

Etape 6 - Nettoyage + documentation
Actions
- Ajouter .gitignore (data/, __pycache__, *.xlsx, *.json exports)
- Mettre a jour README (nouvelles commandes)
- Verifier absence de secrets dans le repo
Checks pour valider
- [x] README a jour
- [x] .gitignore en place
- [ ] Aucun secret commite

Backlog restant (a faire)
- Verifier et supprimer/archiver connection_token/auth.py et connection_token/server.py si obsoletes
- Migrer completement les dashboards dans src/dashboards (pas seulement wrappers) si souhaite
- Valider l'execution end-to-end des scripts V2

Notes
- A chaque etape, si un check echoue, on revient corriger avant de continuer.
- On garde les anciens fichiers tant que la nouvelle version n'est pas validee.
