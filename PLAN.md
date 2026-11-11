# Plan — Quantifier mélange et entrelacement dans les nids de diamants mandarins

## Contexte

- Espèce : diamant mandarin (*Taeniopygia guttata*), nid en coupe dans un support en bois.
- Ficelle fournie **séquentiellement** : orange → jaune → blanc → rouge (rouge à la ponte du 1er œuf).
  La couleur est donc un **marqueur temporel** de la phase de construction.
- 50 nids × 4 vues (top, front, left, right), Canon EOS 550D, mire (échelle + pastille couleur) dans chaque photo.
- Pas de nouvelle collecte prévue : on travaille avec le protocole existant (focale, exposition et distance variables).

## S0 — Données et environnement

- S0.1 Récupérer les 200 images (et la pleine résolution / RAW si elles existent) dans `data/raw/` (lecture seule).
- S0.2 Table de métadonnées : nid, vue, date, couple, EXIF.
- S0.3 Environnement conda (numpy, scikit-image, opencv, scikit-learn, pandas).

## S1 — Prétraitement

- S1.1 Détection de la mire → échelle (px/mm) et correction de balance des blancs.
- S1.2 Masque du nid : retirer fond, support bois, carton, grille, mire, œufs.
- S1.3 Contrôle visuel des masques (planche de vignettes).

## S2 — Carte des couleurs

- S2.1 Annoter quelques zones par couleur (orange, jaune, blanc, rouge, ombre/vide).
- S2.2 Classifieur pixel (features couleur Lab + texture locale), plus robuste que k-means pour le blanc.
- S2.3 Validation sur zones annotées à part (matrice de confusion par couleur).
- S2.4 Nettoyage par continuité : un îlot de couleur doit être « expliqué » — soit il touche d'autres
  brins (morceau visible d'un brin qui passe dessous), soit c'est une extrémité de brin allongée, soit il est
  aligné avec un autre morceau de même couleur de l'autre côté d'un trou (on comble alors le trou le long de
  l'orientation du brin). Sinon c'est un artefact. Version simple ici, version complète en S6.3.

## S3 — Composition

- S3.1 Proportion de chaque couleur par vue et par nid.
- S3.2 Nombre de couleurs utilisées, régularité (evenness).

## S4 — Configuration et mélange

- S4.1 Contacts hétéro/homo-couleur entre pixels voisins (join-count), observé vs attendu à composition égale.
- S4.2 Indice de mélange multi-échelle (entropie en fonction de la taille de fenêtre, normalisée par la largeur d'un brin).
- S4.3 Matrice de contacts entre paires de couleurs : phases adjacentes (orange–jaune) vs éloignées (orange–blanc).

## S5 — Stratification (couleur = temps)

- S5.1 Position de chaque couleur : centre de coupe vs périphérie (vue top), haut vs bas (vues latérales).
- S5.2 Indice de stratification : à quel point l'ordre de fourniture se lit dans la structure spatiale.

## S6 — Segmentation des brins

- S6.1 Détection des structures curvilignes (ridge filter, champ d'orientation), squelettisation.
- S6.2 Détection des croisements et des extrémités.
- S6.3 Reconstitution des brins par continuité d'orientation et de couleur.
- S6.4 Modèle nul au niveau brin : permutation des couleurs entre brins, géométrie conservée (reprend S4.1).

## S7 — Dessus/dessous par la couleur

- S7.1 Aux croisements hétéro-couleur, la couleur visible au centre indique le brin du dessus.
- S7.2 Indice de tissage : taux d'alternance dessus/dessous le long d'un brin.
- S7.3 **Inversions temporelles** : un brin d'une phase tardive passé *sous* un brin plus ancien indique une insertion active, pas un simple dépôt.

## S8 — Analyses

- S8.1 Table finale : une ligne par nid × vue, toutes les métriques.
- S8.2 Cohérence entre vues d'un même nid, corrélations entre indices.
- S8.3 Liens avec les variables biologiques (couple, expérience, succès…) : modèles mixtes.
- S8.4 (Optionnel) Texture à la Bailey et al. (2015) si on cherche une signature individuelle.

## Avancement

Données de test : 8 images (nid 207 × 4 vues ; 107, 206, 208, 308 vue top). La vue **207 left** n'est jamais
annotée : c'est l'image de test hors entraînement.

### 2026-10-23 — S0.3 : environnement
- Env conda `nest` (python 3.12, numpy, scipy, scikit-image, scikit-learn, opencv, pandas, matplotlib).
- 2026-11-04 : l'environnement n'était décrit dans aucun fichier → ajout de `environment.yml` (canal
  conda-forge, versions fixées sur celles utilisées, + imageio et joblib importés directement par le code).
  Pour le recréer : `conda env create -f environment.yml`.

### 2026-10-23 — S2 : classification des pixels par couleur (`scripts/pixel_classify.py`, `src/nestweave/pixels.py`)

Observations préalables :
- La teinte (Lab) sépare bien les couleurs saturées : jaune ≈ 80°, orange ≈ 50°, rouge ≈ 20°.
- Blanc, fond, bois et carton se chevauchent en couleur → il faut de la texture.
- Le rouge du 206 est plus foncé (brique) que celui du 107/308 (rose cramoisi) : autre ficelle ou autre éclairage ?

Versions successives :
- **v1 — un seul random forest** (descripteurs couleur + texture multi-échelle σ 1–8, 6 classes), annotations
  par rectangles + filtre couleur par classe (`annotations/pixel_rects.json`).
  Erreurs : brins fondus en grosses taches (trop de contexte) ; jaune absent du 308 (classé rouge) ;
  blanc du 308 classé « support » ; jaune à l'ombre classé orange en vue latérale.
  L'OOB = 1.0 ne voulait rien dire (pixels d'entraînement trop faciles) → jugé uniquement à l'œil.
- **v2 — deux étages** : (A) masque brin / fond / support avec contexte ; (B) couleur des seuls pixels de
  brin avec descripteurs quasi ponctuels (σ ≤ 1). Ajout d'une balance des blancs sur le fond (expositions
  et réglages variables d'une photo à l'autre). Annotations ajoutées sur 308, 208, 206.
  → le 308 retrouve jaune (10 %) et blanc, brins distincts.
  Erreur : trous sombres entre brins rouges comptés rouges.
- **v2b — classe « ombre »** : pixel de brin plus sombre que 0,6 × la luminance médiane de sa couleur dans
  l'image. Un seuil fixe ne marchait pas (le rouge est naturellement plus sombre que le jaune).
  Constat : les « creux » du 207 top sont en fait du jaune à l'ombre (L > 40), donc le jaune dense est réel.
- **v3a — faux blanc (fond vu à travers les trous)** : le 107, entièrement rouge, avait ≈ 7 % de « blanc ».
  Diagnostic : la ficelle blanche est crème (b ≈ +5 à +20), le fond est neutre (b ≈ 0) une fois la balance
  des blancs faite. Mais une règle de couleur seule (b < 4 ou a > 4) supprimait 68 % des vrais blancs du 308,
  rosis par les poils de ficelle rouge → **rejetée**.
  Solution retenue : nids sans ficelle blanche (107, 206) utilisés comme **exemples négatifs** (tout pixel
  blanchâtre = fond ; `"filter": "white"` dans les annotations). Test croisé (négatifs d'un nid non vus) :
  faux blanc 7 % → 1 % (107), 8 % → 4 % (206). Les vrais blancs du 308 sont conservés (vérifié visuellement :
  ce qui disparaît est la frange en périphérie).
  **Pour les 50 nids : repérer les nids sans blanc et les ajouter en négatifs.**
- **v3b — bug de la règle d'ombre** : avec une couleur presque absente (blanc du 107), la médiane par image
  s'effondrait et des trous sombres devenaient « blanc ». Corrigé : luminance de référence par couleur apprise
  sur l'ensemble des pixels d'entraînement (blanc 71, jaune 60, orange 39, rouge 35).
- Affichage : fond de la segmentation passé en gris-bleu (le blanc-gris se confondait avec la ficelle blanche).
- **v3c — brins devant la boîte** (signalé par Renaud sur le 207 front) : presque tous les brins passant
  devant la boîte (pas seulement les orange) étaient classés « support ». Cause : l'étage A s'appuie sur le
  contexte et n'avait aucun exemple de brin devant la boîte. Une règle « saturé = ficelle » a été écartée
  (le carton du 206 a chroma 36 et une teinte orangée ; la boîte a chroma ≤ 16).
  Solution : rectangles sur la boîte (front, right) filtrés par couleur → seuls les pixels de ficelle y sont
  étiquetés. Corrigé aussi sur la vue left (test hors entraînement).
  Résultat archivé : `outputs/seg_overview_v3_S2.jpg` (= `outputs/seg_overview.jpg`), `outputs/seg_v3_S2/`.

### 2026-10-23 — S2.4 : nettoyage par continuité (`scripts/clean_labels.py`, `src/nestweave/continuity.py`)

Idée (Renaud) : un fil est continu, un îlot isolé est suspect. Nuance : sur la photo, un morceau de brin qui
passe sous d'autres est un vrai îlot (et c'est l'info utile pour S7). Règle retenue : un îlot doit être
« expliqué ».
- Largeur de brin w = 10 px à demi-résolution (≈ 20 px en pleine résolution), estimée sur les squelettes
  des zones où les brins sont bien visibles (107 rouge, 208 blanc). À recalculer en mm quand la mire sera
  détectée.
- **Suppression des îlots non expliqués** : chaque composante d'une couleur reçoit la couleur majoritaire
  de son entourage si elle est minuscule (< 0,5 w²), trop fine (épaisseur < 0,35 w : liserés) ou
  inexpliquée (ne touche aucun autre brin ET n'est pas allongée). Elle est gardée si elle touche un brin
  d'une autre couleur (morceau d'un brin qui passe dessous) ou si elle est allongée (≥ 2,5 w de long,
  allongement ≥ 2,5 : brin ou extrémité de brin).
- **Comblement des trous le long des brins** : fermeture morphologique par segments orientés (12 directions,
  longueur 2 w), acceptée seulement dans la direction locale du brin (tenseur de structure) et seulement
  sur des pixels « support ». Un pixel réclamé par deux couleurs reste tel quel.
- Essai 1 (rejeté) : comblement aussi sur les pixels « ombre » avec longueur 2 w. Erreur : les vrais trous
  entre brins (308, 207 top) étaient remplis par la couleur voisine, ce qui rend les zones de même couleur
  artificiellement continues → **biais vers moins de mélange**, exactement ce qu'on veut mesurer.
- Version retenue : support comblé sur 2 w ; ombre comblée seulement sur des trous courts (< 1 w, dans l'axe
  du brin : creux de la torsade de la ficelle, sombres sur la boîte sombre). Les pixels d'ombre baissent de
  30 à 60 % selon l'image ; les grands trous du 308 restent noirs. **Réglage à surveiller** (`gap_frac`).
- Effet : 800 à 1 700 mouchetures retirées par image, 1,5 à 3 % des pixels modifiés, proportions de couleurs
  quasi inchangées (± 0,01). Brins devant la boîte plus continus (207 front), une partie des liserés orange
  le long du jaune retirés.
  Résultat : `outputs/seg_clean/`, `outputs/seg_overview_v4_S2.4.jpg` (à comparer avec v3).
- Erreur révélée par S2.4 (206) : une bande orange continue sur le support en bois. Elle venait de S2 (couche
  inférieure brun-orangé du contreplaqué classée orange, mes rectangles « support » ne couvraient que la
  partie claire du bois) ; S2.4 l'avait juste rendue continue. Corrigé en ajoutant 2 rectangles « support »
  sur la bande sombre, ré-entraînement, puis S2 + S2.4 relancés (`outputs/seg/` et `outputs/seg_clean/`
  régénérés ; v3 archivée inchangée dans `outputs/seg_v3_S2/`).
  Leçon : S2.4 amplifie les erreurs de S2 qui ont une forme allongée → toujours comparer avant/après.
- **Validé par Renaud** (v4 retenue comme segmentation de travail).

### 2026-10-23 — Nouvelles images `all_views/` (28 images, 4 nids)
- 102 (2 dates × 5 vues), 109 (5 vues), 207 (4 vues, identiques à `data/`), 209 (14/02 : 5 vues ; 15/08 : 4 vues).
  Nouvelle vue **entrance** (oblique, bord noir de la boîte à lumière dans le cadre).
- 102 : 14/04 sans rouge, 28/04 avec rouge par-dessus → **confirmé par Renaud : même nid, avant/après
  l'apport du rouge (ponte)**.
  209 : 14/02 jaune/orange, 15/08 avec beaucoup de blanc, sans rouge → Renaud a répondu « oui » à une
  question à deux options (même nid 6 mois plus tard / nouveau nid du même couple) : **à préciser**.
- Segmentation S2 + S2.4 avec le modèle actuel, **aucune annotation sur ces images** (test de généralisation) :
  `outputs/all_views/seg/`, `outputs/all_views/seg_clean/`.
  Rouge du 102 : 0 % le 14/04 → 40–60 % le 28/04. Blanc du 209 : ≈ 5 % → 40–60 %.
  Erreur suspectée : **faux blanc sur les nids jaunes** (109, entièrement jaune : 4–14 % de « blanc ») ;
  les exemples négatifs ne venaient que de nids rouges. **Vérifié visuellement** : c'est un liseré blanc
  le long du contour du nid (pixels de bord qui mélangent jaune et mur gris → couleur crème), visible aussi
  sur 102 top 28/04. Piste de correction générale (sans annoter chaque nid) : un pixel « blanc » à moins
  de ~3 px à la fois du fond et d'un brin coloré est un pixel de mélange → fond. Un vrai brin blanc
  (≈ 10 px de large) ne perd que ses bords.

### 2026-10-23 — Faux blanc sur les nids jaunes (v5, v5b)
- Renaud signale le même problème sur le 207 right.
- Zoom : la piste « pixel de mélange » ne marche pas, il y a 2 mécanismes différents :
  (a) 109 : **trous entre brins jaunes** où l'on voit le mur, entièrement classés blancs (aucun pixel
  « fond » autour, donc la règle ne s'appliquerait pas) ; (b) 207 right : vrais brins blancs mais le blanc
  **déborde dans le fond** autour des fibres (pas de brin coloré à côté). Piste abandonnée avant codage.
- La couleur ne sépare pas : le mur vu à travers les trous du 109 est teinté de jaune (b médian ≈ 9),
  proche du vrai blanc crème (b ≈ 13–15).
- **v5** : exemples négatifs sur des nids jaunes sans blanc (109 × 5 vues, 102 du 14/04 × 5 vues), sans
  rectangles manuels : filtre `white_near_colour` = pixel blanchâtre à < 2 largeurs de brin (20 px) d'une
  ficelle saturée → fond. Clés d'annotation passées en chemins relatifs à la racine (`data/…`, `all_views/…`).
  Faux blanc sur nids testés sans blanc : 209 du 14/02 4 % → 0 %, 102 du 28/04 → ≈ 0.
  **Erreur : sur-correction**, les vrais blancs disparaissent aussi (207 left 10 % → 2 %, 209 left 15/08
  51 % → 35 %, 207 right 24 % → 10 %), vérifié visuellement. Cause : déséquilibre, ≈ 200 000 pixels
  « blanchâtre près de la couleur = fond » contre ≈ 60 000 pixels de vrai blanc.
- **v5b** : mêmes négatifs plafonnés à 3 000 pixels par image (`"max_px"`). Faux blanc toujours ≈ 0 sur les
  nids testés sans blanc ; vrais blancs : 207 left 10 → 4 %, 207 right 24 → 15 %, 209 du 15/08 −4 à −10
  points, 208 84 → 83 %. Visuellement : franges et débordements réduits (mieux), mais une partie des vrais
  brins blancs fins en périphérie est perdue (207 left surtout). **Compromis : le blanc reste la classe la
  moins fiable, probablement sous-estimé en v5b et surestimé en v4.**
- Remarque : les luminances de référence (seuil d'ombre) avaient changé silencieusement depuis v3c
  (jaune 60 → 51, orange 39 → 30) à cause des brins à l'ombre devant la boîte ajoutés à l'entraînement.
- **Décision (Renaud) : on avance avec v5b, mais on devra sans doute revenir sur la segmentation plus
  tard** (blanc en priorité ; pistes : pleine résolution, plus d'exemples de vrai blanc, validation S2.3).
  Les métriques sont calculées sur v4 ET v5b pour mesurer la sensibilité des conclusions au blanc.
- Archives : `outputs/seg_v4_S2.4/`, `outputs/seg_v5b/`, `outputs/all_views/seg_v4_S2.4/`,
  `outputs/all_views/seg_v5b/` ; planches `outputs/seg_overview_v5b.jpg`,
  `outputs/all_views/overview_v4.jpg`, `outputs/all_views/overview_v5b.jpg`.

### Note de méthode — d'où vient l'« entraînement »
Les images ne sont pas annotées par l'équipe : les annotations sont des rectangles tracés par Claude à l'œil
(≈ 60 rectangles sur 7 images, `annotations/pixel_rects.json`), nettoyés par un filtre couleur par classe.
Limites : annotateur non expert, circularité partielle pour les couleurs (les filtres définissent le
« jaune »), validation uniquement visuelle jusqu'ici. **Avant publication : validation indépendante (S2.3)**,
par ex. 100–200 points tirés au hasard par image sur ~20 images, classés par un humain sans voir la
segmentation → matrice de confusion.

Défauts restants après S2.4 :
- bord ombré des brins jaunes classé orange : en partie seulement corrigé par S2.4 ; risque de gonfler les
  contacts hétéro-couleur (S4) ;
- le blanc à l'ombre au fond de la coupe est classé « support » (207 front) ; brin beige-gris devant la boîte
  en partie raté ;
- extrémités fines de brins rouges sur fond blanc classées « support » (206, bord gauche) ;
- halo « support » autour du nid ;
- mire et bois pas encore détectés automatiquement (pas encore d'échelle px/mm).

### 2026-10-23 — S3 / S4 / S5 : premières métriques (`src/nestweave/metrics.py`, `scripts/metrics.py`, `scripts/plot_metrics.py`)

Méthode (unité de distance : largeur de brin w = 10 px) :
- **S3 composition** : part de chaque couleur parmi les pixels de brin ; couleur < 2 % considérée absente
  (bruit) ; richesse, Shannon, régularité (evenness), hétérogénéité de Simpson 1 − Σp².
- **S4 indice de mélange** (courbe) : fraction de paires de pixels de brin de couleurs différentes à une
  distance donnée (1, 2, 4, 8, 16 w ; 8 directions), divisée par 1 − Σp² (attendu si les couleurs étaient
  placées au hasard à composition égale). ≈ 0 = blocs de couleur plus grands que l'échelle, ≈ 1 = hasard.
  Distance minimale = 1 w et pas pixels voisins, pour que les liserés de 1–2 px ne créent pas de faux
  contacts. Aussi : rapport observé/attendu par paire de couleurs à 1 w, et entropie moyenne dans des
  fenêtres de 2 à 32 w (approche 1 du document de projet).
- **S5 stratification** : vues latérales, hauteur moyenne de chaque couleur (0 = bas, 1 = haut de
  l'étendue des pixels de brin) ; vues top/entrance, centralité. ρ = corrélation de Spearman entre l'ordre
  d'apport (orange 1, jaune 2, blanc 3, rouge 4) et la position (> 0 = couleurs tardives plus haut / plus
  au centre).
- Tests sur images synthétiques avant usage : deux moitiés → mélange ≈ 0,03 à 1 w ; blocs aléatoires de
  1 w → ≈ 1,0 ; couches empilées → ρ = 0,94 ; une seule couleur → non défini. OK.

Résultats (32 images, 9 échantillons nid × date ; v5b ; `outputs/metrics_v5b.csv`, `outputs/metrics_v4.csv`,
figures `outputs/figs/`) :
- **Mélange à l'échelle du brin (1 w)** : 207 le plus groupé (0,25–0,34 selon la vue), 308 / 206 /
  102 du 28/04 les plus mélangés (≈ 0,5–0,6). L'ordre correspond à l'impression visuelle (207 « couleurs
  groupées », 308 « très mélangé »).
- **Forte dépendance à la vue** pour certains nids (102 du 14/04 : entrance 0,75 vs right 0,26), faible pour
  d'autres (207, 209 du 15/08, 102 du 28/04) → il faudra plusieurs vues par nid (effet aléatoire vue).
- **Indice instable quand une couleur est très minoritaire** (109 : orange 2–9 %) : valeurs > 1 aux
  grandes échelles. À filtrer (seuil d'evenness ou intervalle de confiance) avant toute analyse.
- **Stratification** : en moyenne sur les 3 vues latérales de chaque échantillon, l'orange (apporté en 1er)
  est le plus bas ou à égalité (102 du 28/04 : à égalité avec le rouge), le jaune au-dessus, le blanc en
  haut ou à égalité avec le jaune (207 : ρ = 0,60 ; 102 avant rouge : 0,53 ; 209 du 15/08 : 0,38).
  L'ordre d'apport se lit donc dans la structure verticale sur plusieurs nids, mais pas partout : ρ est
  faible pour 109 (0,18) et 209 du 14/02 (0,20), et négatif sur certaines vues isolées (209 du 14/02 left
  −0,13 ; 102 du 28/04 left −0,11).
  *Correction du 2026-10-30 : la première rédaction disait « dans toutes les vues latérales », ce qui était
  trop fort.*
- **102 avant/après rouge** : le rouge (28/04) n'est PAS au-dessus — même hauteur moyenne que l'orange
  (≈ 0,65), ρ tombe de 0,53 à 0,08 ; le mélange à 1 w augmente (vues latérales 0,26–0,47 → 0,47–0,59).
  Mais **attention** : la hauteur est normalisée par l'étendue des brins, qui change quand des brins rouges
  pendent plus bas (l'orange « remonte » de 0,36 à 0,65 sans avoir bougé) → il faut une référence fixe
  (la boîte / le support, S1) avant d'interpréter.
- **Sensibilité v4 vs v5b** (`outputs/figs/sensitivity.png`) : la part de blanc change (v5b < v4), mais
  l'indice de mélange (écarts ≤ ≈ 0,1) et ρ restent proches de la diagonale → les conclusions ci-dessus ne
  dépendent pas de l'incertitude sur le blanc, pour ces échantillons.
- Limites : 6 nids, aucune statistique possible ; pixels ≠ brins (un long brin pèse plus qu'un court) ;
  surface visible seulement.

### 2026-11-10 — S1.1 : échelle par la mire (`src/nestweave/reference.py`, `scripts/detect_scale.py`)
- Résultat : `outputs/scale.csv` (colonne `px_per_mm_used`), contrôle visuel `outputs/scale_check.jpg`.
  **31 images / 32 ont une échelle** : 30 par la mire, 1 par la médiane de sa séance (102 left 28/04).
  Seule **308 top** n'a rien (règle entièrement cachée par la ficelle).
- Principe : cases noires de la règle (rectangulaires ~2:1, une rangée = 10 mm) → toutes les paires de
  cases proposent un axe et un pas → on garde l'hypothèse qui aligne le plus de rangées → moindres carrés.
- Erreurs et corrections, dans l'ordre :
  1. Hypothèse de cases carrées → échec partout. Les cases sont rectangulaires → corrigé (vues top seulement).
  2. Vues de côté : cases collées par les coins au seuil 75 → seuils multiples (40–85) et deux ouvertures,
     candidats mis en commun. Les vues de côté passent, mais **faux positifs** : trous sombres entre brins
     de ficelle rouge pris pour des cases (107, 206, 308, 102 left 28/04 : valeurs 3,7–4,5 fausses).
  3. Filtre « carte autour » : case grise ET anneau autour clair et peu saturé → trop strict (sur les vues
     de côté la carte est à l'ombre, blanc ≈ gris 110) → on ne garde que la **saturation** (vraies cases ≈ 10–20,
     trous dans le rouge ≈ 120–140).
  4. 206 (surexposée : cases noires à gris ≈ 100) et 102 left 28/04 (carte à l'ombre) non détectées →
     ajout d'un **seuil adaptatif** (case plus sombre que son voisinage). 206 retrouvée (6,33). 102 left 28/04
     toujours non : cases collées aux graduations et au bord de la carte.
  5. Choix d'hypothèse trop tolérant (107 : 9 rangées dont le bord noir de la carte) → affinage (rejet des
     cases à > 0,08 rangée, réajustement), test d'**orientation** (grand côté de la case parallèle à la
     règle), et **au plus 8 rangées** (règle 0–80 mm).
  6. Constat : sur une séance, toutes les vues de côté donnent la même échelle à ~1 % (207 : 6,01 ;
     102 du 14/04 : 5,57 ; 109 : 5,58 ; 209 du 14/02 : 5,60) → l'appareil est à distance fixe.
     J'ai d'abord remplacé toute vue s'écartant de > 3 % de la médiane de séance → **erreur** : les deux vues
     d'entrée 102 du 14/04 et 209 du 14/02 (3,2–3,3 px/mm) sont vraiment prises plus dézoomées (coins noirs
     sur les photos ; vérifié à la main sur les deux moitiés de la carte : ≈ 3,23 et ≈ 3,3 px/mm).
     → règle finale : médiane de séance **seulement si aucune mire n'est détectée** ; un écart est signalé
     (`source = ruler_off_session`) mais conservé.
- Échelles obtenues : vues top 6,2–7,5 px/mm (varie d'une séance à l'autre) ; vues de côté 5,3–6,0 ;
  2 vues d'entrée à 3,2–3,3. Un brin (w = 10 px à demi-résolution = 20 px) fait donc ≈ 3–3,5 mm.
- Limite : la mire n'est pas dans le plan du nid (elle est devant/à côté) → l'échelle au niveau du nid peut
  différer de quelques % ; suffisant pour comparer, pas pour des mesures absolues fines.
- Conséquence pour S4 : les échelles de l'indice de mélange (lags en pixels) ne sont pas les mêmes d'une
  vue à l'autre (facteur jusqu'à ~2 pour les vues d'entrée dézoomées) → à convertir en mm.
- Reste à faire en S1 : la référence fixe (boîte / support) pour renormaliser la stratification.

### 2026-11-10 — S1 : référence de hauteur fixe = la grille verte (`reference.cage_bars`)
- La boîte est posée sur une grille métallique vert foncé à 3 barres horizontales. Aucune ficelle n'est
  verte → masque vert (teinte 55–100, S > 35, 30 < V < 150), plus gros amas = la grille, barres = pics du
  profil vertical du masque. Zéro = **barre du haut** (là où repose la boîte).
- Erreurs et corrections :
  1. Ouverture morphologique horizontale (150 px) pour trouver les barres → rien : les fils sont fins et
     interrompus par la ficelle → profil ligne par ligne + détection de pics.
  2. Coins noirs des photos dézoomées : bruit de teinte aléatoire pris pour du vert → exclusion des zones
     noires (V < 40) élargies.
- Contrôles : hauteur barre du haut → barre du bas = **40,5–43,1 mm** sur les 11 vues où les 3 barres sont
  vues (mire × grille : deux mesures indépendantes qui concordent). Dans une séance, la barre du haut est
  à la même ligne d'image (±10 px) → repli sur la médiane de séance si absente ou à > 40 px.
- Résultat (`outputs/scale.csv`, `cage_top_y_used`) : toutes les vues de côté ont un zéro, sauf
  102 entrée du 14/04 (dézoomée, grille non trouvée ; les vues d'entrée ne servent pas à la hauteur).
- `metrics.height_mm` + `scripts/metrics.py --refs` → colonnes `hmm_<couleur>` (médiane en mm au-dessus de
  la grille, vues front/left/right) dans **`outputs/metrics_v5b_S1.csv`** (`metrics_v5b.csv` conservé).
  ρ de stratification inchangé (il ne dépend que de l'ordre des hauteurs).
- **102 avant/après rouge, refait avec le zéro fixe** (médianes en mm, front / left / right) :
  - jaune : 88 / 75 / 97 → 91 / 70 / 86 (≈ stable) ;
  - orange : 58 / 32 / 53 → 63 / 58 / 74 (**monte réellement** de 5 à 26 mm, ce n'est plus un artefact de
    normalisation comme je le supposais) ;
  - rouge (28/04) : 80 / 55 / 84 → **entre l'orange et le jaune, pas au-dessus** ;
  - sommet du nid (98e centile) : 125–129 → 110–114 mm (≈ 15 mm plus bas) ; bas des brins pendants :
    de −10…+8 à −51…−14 mm (le rouge pend plus bas).
  - Lecture probable : le rouge est ajouté sur les côtés et pend, il **recouvre l'orange bas** → l'orange
    visible qui reste est plus haut. Le nid s'est aussi tassé ou a été remanié (sommet −15 mm).
    Limite : on ne mesure que la surface visible ; « l'orange monte » peut être « l'orange bas est caché ».

### 2026-11-10 — S4 : indice de mélange en mm (`metrics.mixing_curve_mm`)
- Distances fixées en mm (3, 6, 12, 24, 48 ; un brin ≈ 3,2 mm) au lieu de multiples de w = 10 px,
  avec l'échelle S1 de chaque image → colonnes `mix_<d>mm` de `outputs/metrics_v5b_S1.csv`.
- 308 top (règle cachée) : échelle = médiane des vues top (6,59 px/mm, `source = top_median`), précise à
  ~±10 % seulement (les vues top vont de 6,2 à 7,5).
- Effet : les 2 vues d'entrée dézoomées étaient biaisées vers « plus mélangé » (à 10 px on comparait des
  points 2× plus éloignés) : 102 entrée 14/04 0,75 → **0,47** à 3 mm ; 209 entrée 14/02 0,58 → **0,36**.
  Les autres vues bougent peu (10 px demi-résolution ≈ 3,6 mm, proche de 3 mm). Classement inchangé :
  207 reste le moins mélangé.
- Figure : `outputs/figs/mixing_curves.png` maintenant en mm ; l'ancienne version en px est gardée sous
  `outputs/figs/mixing_curves_v5b_px.png`.

### 2026-11-10 — S2.3 : outil de validation à l'aveugle (`scripts/make_validation.py`, `scripts/eval_validation.py`)
- Je ne peux pas annoter moi-même : la vérité terrain doit être indépendante du modèle (et de moi).
- 367 points tirés (graine 0) sur la segmentation v5b, **hors de tous les rectangles d'entraînement** :
  - 192 « uniformes » (6 par image, dans la zone du nid élargie de ~20 px) → exactitude et rappel sans biais ;
  - 175 « stratifiés » (25 par classe prédite) → précision de chaque classe, y compris le blanc (rare).
  - Limite : les entrées d'entraînement sans rectangle (négatifs « blanc près de couleur » de 109/102,
    3000 px tirés au hasard) ne peuvent pas être exclues ; recouvrement négligeable.
- Page `validation/index.html` (s'ouvre dans un navigateur, hors ligne) : vignette de contexte + zoom ×5
  avec viseur, **sans la prédiction**, 8 réponses (blanc, jaune, orange, rouge, trou/ombre, fond, autre,
  incertain), raccourcis 1–8, sauvegarde dans le navigateur, bouton « Exporter » → `annotations_<nom>.json`.
- `eval_validation.py` : fond + autre fusionnés en « non-ficelle » (les métriques n'utilisent pas la
  différence) ; exactitude globale et rappel sur l'échantillon uniforme, précision par classe prédite sur
  tous les points, IC95 de Wilson ; accord entre annotateurs si plusieurs fichiers.
- Testé avec des réponses simulées (non conservées). **En attente des annotations.**
- 2026-11-11 — **bug signalé par Renaud** (« le centre du viseur semble décalé entre contexte et zoom ») :
  le viseur était centré sur le cadre image + légende au lieu de l'image seule → affiché ~10 px écran trop
  bas, soit ≈ 12 px de la photo sur le contexte (≈ ½ largeur de brin) et ≈ 2 px sur le zoom. Les
  vignettes elles-mêmes étaient justes (point au centre à < 1 px). Corrigé (viseur dans un cadre limité à
  l'image) ; page régénérée avec les **mêmes points** ; clé de sauvegarde passée en v2 → les réponses
  données avec l'ancien viseur ne sont pas rechargées (elles visaient le mauvais endroit). Consigne
  ajoutée : sur une limite, le zoom fait foi.

## Questions ouvertes

- Quantité de ficelle fournie par phase : identique pour chaque couleur ?
- Numérotation des nids (107, 206, 308…) : couple, pièce, volière ?
- Séries temporelles après le 1er œuf : elles couvrent la phase rouge, donc peut-être utiles pour voir où le rouge est ajouté.
