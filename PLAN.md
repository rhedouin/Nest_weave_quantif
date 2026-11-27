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
- 2026-11-11 — question de Renaud sur les cas limites trou/ombre vs couleur (points 58, 59, 64, 68, 83,
  102). Règle fixée : **la couleur dès qu'elle se reconnaît, même assombrie** ; trou/ombre seulement
  s'il n'y a aucun brin identifiable au centre ; le duvet compte pour son brin s'il en montre la couleur ;
  vrai 50/50 → incertain. Raison : les métriques n'utilisent que les pixels de couleur ; ce qu'on veut
  savoir, c'est si le modèle perd ou se trompe de couleur. Ajouté à la page et à l'évaluation : score
  « couleur juste » (vérité et prédiction toutes deux des couleurs) et taux de vraies couleurs prédites
  trou/ombre (pixels perdus, pas mal étiquetés).

 ### 2026-11-17 [MD]- annotations et commentaires de Marie 
  - Annotations from validation html document done by marie. annotations_MD.json fichier ajouté au dossier annotations 
  - QUESTIONS & COMMENTAIRES
    - qu'est-ce qui dans le fonctionnement du pipeline fait que les extremités des ficelles sont souvent "oubliées" meme quand la couleur semble simple a distinguer (ex: `outputs\seg_v5b\107_top_20260526_overlay.jpg`) ? 
    - c'est quoi les fichiers type `outputs\seg\107_top_20260526_labels.png` je distingue vaguement une ombre grise sur fond noir ? 
    - je suggère d'ignorer les "entrance views" pour tout ce projet. 

### 2026-11-19 — Fusion des branches de Marie + S2.3 : résultats de la validation (segmentation v5b)
- Branches `origin/marie` (annotations, calendrier d'apport `docs/string_provisionning_schedule.pdf`,
  commentaires ci-dessus) et `origin/marie-plan-edits` (réponses aux questions ouvertes) fusionnées
  dans `master`, sans conflit.
- `python scripts/eval_validation.py annotations/annotations_MD.json` — 367 points annotés par Marie,
  19 « incertain » écartés. (Fichier rangé dans `annotations/` au lieu de `validation/` : sans effet.)
- **Exactitude globale 0,80** [IC95 0,74–0,85] (échantillon uniforme, n = 184) ; sur la ficelle seule 0,68.
- **Quand le modèle et l'annotatrice donnent tous deux une couleur, elle est juste à 0,91.** Les erreurs
  viennent surtout d'ailleurs :
  1. **Ficelle manquée (prédite non-ficelle)** : 14 % des vrais points de ficelle de l'échantillon
     uniforme (blanc 5/17, orange 5/18, jaune 3/50, rouge 1/10). Sur les 15 cas (tous points) : la moitié
     à ≤ 6 px du bord d'un brin détecté (bords), l'autre moitié à 14–39 px = **bouts effilochés isolés**,
     surtout sur le mur (107, 207) ou devant le bois. **14 sur 15 sont déjà perdus à l'étape A
     (masque ficelle / non-ficelle)**, 1 seul par le nettoyage S2.4.
     → la part de **blanc est sous-estimée** (rappel 0,59) : fibres blanches fines sur fond blanc.
  2. **Jaune à l'ombre pris pour de l'orange** : 8 cas ; précision de l'orange 0,62 seulement
     (16 erreurs sur 42 : 8 jaune, 4 trou/ombre, 5 non-ficelle). Le jaune assombri vire vers la teinte
     orange. Effet probable : contacts jaune–orange fictifs → **mélange surestimé** sur les nids jaunes
     (109, 102 du 14/04, 209 du 14/02).
  3. **Classe « ombre » du modèle** : précision 0,29 — elle contient surtout de la ficelle dont la couleur
     reste reconnaissable à l'œil (jaune 7, rouge 4, orange 4, blanc 3). Peu grave pour les métriques
     (pixels perdus, pas mal étiquetés : 3 % des vraies couleurs), mais l'ombre est trop large.
- Précision des autres couleurs bonne : blanc 0,94, jaune 0,95, rouge 0,88.
- Pistes, par ordre de gain attendu : (a) étape A à pleine résolution et/ou avec moins de poids aux
  grandes échelles (σ = 4–8 noient les bouts fins dans le fond) ; (b) caractéristiques de couleur
  normalisées par la luminance (chromaticité) pour séparer jaune sombre / orange ; (c) seuil d'ombre plus
  bas (0,6 → ~0,45 × L de référence). Chaque changement devra être revalidé sur ces mêmes 367 points.

### 2026-11-19 — Réponses aux commentaires de Marie
- **Bouts de ficelle « oubliés »** (ex. 107 top) : la validation le confirme et le localise : c'est
  l'étape A (masque ficelle / non-ficelle) qui les perd, pas le nettoyage. Elle travaille à demi-résolution
  et avec des descripteurs lissés jusqu'à σ = 8 px : autour d'un bout fin isolé sur le mur, le voisinage
  est presque entièrement du fond, donc les descripteurs « disent » fond. Les fibres fines sont aussi
  mêlées au fond par la réduction de résolution. Corrections possibles : voir pistes (a) ci-dessus.
- **Fichiers `*_labels.png`** : ce sont les cartes de classes brutes, un pixel = un numéro de classe
  (1 fond, 2 autre, 3 blanc, 4 jaune, 5 orange, 6 rouge, 7 ombre). Ces valeurs (1 à 7 sur 255) sont
  presque noires à l'écran — d'où l'« ombre grise ». Elles servent aux calculs ; pour regarder, utiliser
  les `*_overlay.jpg` (classes en couleur).
- **Ignorer les vues d'entrée** : d'accord sur le fond — cadrage différent (2 vues dézoomées), carte
  inclinée, pas de référence de hauteur, axe « centralité » discutable. **À valider par Renaud**, puis les
  retirer des figures et des analyses (on peut les garder dans les CSV).
- **Indice de mélange et nombre de couleurs** : l'indice actuel est déjà normalisé par la composition
  observée sur la photo (division par 1 − Σp², la probabilité que deux points au hasard soient de couleurs
  différentes) : un nid avec 2 couleurs n'est pas pénalisé par rapport à un nid avec 3. Il utilise les
  proportions de pixels visibles, pas le nombre de couleurs fournies ; une couleur < 2 % est traitée comme
  absente. Le nombre de couleurs fournies jusque-là (calendrier) peut s'ajouter comme covariable en S8.
- **« Seule la 1re série, sans rouge, analysée d'abord »** : à noter que parmi les données actuelles,
  102 du 28/04, 107, 206 et 308 contiennent du rouge (donc pas de 1re série ?). À préciser avec Marie
  quand les autres images arriveront.

### 2026-11-19 — Rapport LaTeX : évolution de la segmentation (`reports/segmentation_evolution/`)
- Demande de Renaud : visuels avant / après, vues d'ensemble et zooms, utilisables pour un article.
  Précisé avec lui : les annotations de Marie notent la segmentation mais ne la modifient pas → le rapport
  montre l'**évolution v3 → v4 → v5b** + la validation de v5b, sur les images annotées (`old/data`,
  `old/all_views_01_11` ; Renaud a remplacé `all_views/` par un dossier plus récent).
- Figures et tableaux régénérables : `python scripts/make_report_figures.py`, puis `latexmk -pdf main.tex`
  dans `reports/segmentation_evolution/` → `main.pdf` (7 pages).
- Vérifications faites en le préparant : v1/v2 non archivées (comparaison à partir de v3) ; le zoom 207
  front v3/v4 ne montrait presque rien (la correction des brins devant la boîte était déjà dans v3) →
  remplacé par 308 top pour illustrer S2.4 ; sur 206, la bande orange du contreplaqué est **largement**
  mais pas entièrement supprimée en v4.
- Erreurs corrigées dans le texte avant livraison : « 36 photos de 9 nids » → 32 photos (207 en double),
  8 nids ; 184 points uniformes → 192 tirés (184 après retrait des incertains).
- **Attention** : `annotations/pixel_rects.json` pointe vers `data/…` et `all_views/…` ; ces images sont
  maintenant dans `old/`, donc un ré-entraînement échouerait ou utiliserait les mauvaises images tant que
  les chemins ne sont pas mis à jour.

### 2026-11-20 — Décisions de Renaud et réparation de l'entraînement
- Ordre de travail validé : chemins d'annotation → pistes (a) et (b) avec revalidation sur les 367 points
  → passage des 51 nouvelles images (`all_views/`, 9 nids dont 309, 13 séances) → S8.
- **Le rouge est gardé** : on n'analyse pas seulement la 1re série sans rouge, toutes les séries comptent.
- **Vues d'entrée (entrance) ignorées** pour tout le projet (proposition de Marie, validée par Renaud) :
  vue oblique, bord noir de la boîte dans le cadre, 2 sur 4 dézoomées (≈ 3,3 px/mm au lieu de ≈ 5,6),
  pas de repère de hauteur fiable. Gardées dans les CSV, retirées des figures et des analyses.
- `annotations/pixel_rects.json` : chemins `data/…` → `old/data/…`, `all_views/…` → `old/all_views_01_11/…`
  (17 images, même format). Les images de `all_views/` actuel sont identiques (md5) à celles de l'ancien
  dossier, mais on pointe vers `old/` pour figer les images annotées.
- Vérification : ré-entraînement dans un fichier à part (graine fixe) → segmentation brute **identique au
  pixel près** à `outputs/seg/` sur 207 top et 109 right. Le modèle `models/pixel_rf.joblib` n'a pas été touché.

### 2026-11-20 — S2.3 : banc de revalidation et premières variantes
- Outils : `scripts/revalidate.py --model M --out D` (segmente les 32 images annotées + S2.4, lit la classe
  sous les 367 points → `D/predictions.csv`, ~5 min) ; `scripts/compare_variants.py` (une ligne par variante,
  + comparaison **appariée** à v5b sur l'échantillon uniforme : points corrigés / cassés, test de McNemar
  exact) ; `eval_validation.py --pred` pour le détail d'une variante. Paramètres des descripteurs, du
  seuil d'ombre et des filtres de teinte désormais stockés dans le modèle (`pixel_classify.py train
  --sigma-min/--sigma-max/--num-sigma/--chroma-norm/--yellow-min-hue/--orange-max-hue`).
- Contrôle : le modèle actuel passé dans le banc redonne les 367 prédictions v5b à l'identique.
- Limite : les points stratifiés ont été tirés par classe **prédite par v5b** → la précision d'une autre
  variante est calculée sur un échantillon penché vers v5b ; les chiffres de l'échantillon uniforme
  (exactitude, manqués, rappel, corrigés / cassés) ne le sont pas.
- Résultats (corrigés / cassés vs v5b sur 184 points uniformes) — **aucune différence significative** :
  (c) seuil d'ombre 0,45 : 3 / 1, mais la classe ombre disparaît et la précision de l'orange baisse
  (0,62 → 0,57) : le jaune sombre libéré part en orange ; (a1) σ 0,5–4 : 4 / 3, manqués 0,15 → 0,13 ;
  (a2) σ 0,5–8 : 2 / 2 ; (b) couleur normalisée par la luminance : 0 / 2, orange 0,59.
- **Cause trouvée pour jaune → orange** : ce n'est pas un problème de descripteurs mais de données
  d'entraînement. Le jaune à l'ombre a une teinte Lab de 57–66° (jaune bien classé ≈ 82°, orange ≈ 49°) ;
  le filtre des rectangles « jaune » exige teinte ≥ 66°, donc le jaune ombré n'est jamais montré comme
  jaune, et il tombe dans la plage du filtre orange (40–66°). Variante (d) : jaune ≥ 58°, orange < 62°.
- Résultats (d) — v5b → d → d + ombre 0,45 : exactitude 0,80 → 0,82 → **0,84** [0,78–0,88] ; couleur juste
  quand les deux voient une couleur 0,91 → 0,96 → 0,95 ; **précision de l'orange 0,62 → 0,84 → 0,77** ;
  rappel jaune 0,80 → 0,84 → 0,90, rouge 0,80 → 0,80 → 0,90, blanc 0,59 → 0,59 → 0,65 ; jaune → orange
  dans l'échantillon uniforme 5 → 0 → 1, aucun orange → jaune. Apparié : d 5 corrigés / 2 cassés,
  d + ombre 0,45 **10 / 3 (p = 0,09)**. Combinés, les deux changements vont dans le même sens ; seul, le
  seuil d'ombre ne valait rien (le jaune sombre libéré partait en orange).
- Vérif. visuelle : 102 front 28/04, brins rouges à l'ombre (classés « ombre » en v5b) → rouges, juste ;
  209 left 14/02, fausses taches orange dans le jaune supprimées, vraie bande orange gardée. Effet sur les
  proportions : orange −3 à −11 points sur les nids jaunes (109 right 9 % → 0 %), « ombre » divisée par 2 à 5.
- La classe « trou / ombre » ne reconnaît de toute façon jamais les vrais trous (0/10 dans toutes les
  variantes : ils sont prédits non-ficelle) ; elle ne sert qu'à écarter des pixels douteux.
- **Adopté comme v6** : `models/pixel_rf_v6.joblib` = `models/pixel_rf.joblib` (seuil d'ombre 0,45 stocké
  dans le modèle) ; ancien modèle archivé en `models/pixel_rf_v5b.joblib`. Cartes v6 des 32 images
  annotées : `outputs/seg_v6/` (déjà nettoyées S2.4).
- **Réserve importante** : les bornes 58° / 62° ont été choisies en regardant les erreurs de ces mêmes
  367 points → la validation n'est plus tout à fait indépendante pour v6. Pour l'article : tirer un
  **nouveau jeu de points sur les nouvelles images** (`all_views/`, jamais vues) et le faire annoter.
- Non résolu : **ficelle manquée** (14–15 %, blanc surtout). Les descripteurs plus fins (a1, a2) ne gagnent
  que 2 points. Probablement le même type de cause (données) : aucun rectangle d'entraînement ne couvre
  des bouts fins isolés sur le mur. Correction possible : quelques rectangles « blanc » / « orange »
  sur des bouts effilochés (annotation à faire), puis revalidation.

### 2026-11-20 — Nouveau jeu de données complet : analyse (`scripts/inventory.py` → `outputs/inventory.csv`)
- Renaud a reçu toutes les données, classées en `05_Top/` (178), `02_Left/` (176), `03_Right/` (177).
  `all_views/` déplacé dans `old/`. Décision avec Marie : **repartir sur les vues du dessus seulement**.
  Renaud fera lui-même les ré-annotations nécessaires.
- Homogène : Canon EOS 550D, 2592×1728, ISO 100, noms tous conformes `<nid>_<vue>_<date>.JPG`.
- **32 nids, 176 séances du dessus**, du 22/10/2025 au 17/10/2026. Presque toutes les séances ont
  dessus + gauche + droite. Par nid : 2 à 11 photos du dessus ; 16 nids ont 2 épisodes de construction
  (séances groupées, écart > 40 j), 2 en ont 3 → cohérent avec « 2 nids par couple », mais 14 nids n'en ont
  qu'un. Le plus souvent 3–4 séances par épisode.
- Anomalies :
  - **104 et 207 du 07/11, 11/11 et 18/11/2025 : mêmes fichiers sous les deux noms** (9 paires, 3 vues).
    **Résolu (table de Marie)** : en cohorte 1 – ronde 1, les photos portent un **code photo aléatoire à
    3 chiffres**, pas le numéro de paire, et ces codes peuvent coïncider avec des numéros de paire.
    207 et 104 sont les deux codes du nid MIA_10401 (paire 104) → les « 207 » de nov. 2025 ne sont **pas**
    la paire 207. Une seule copie gardée.
  - L'ancien `107_*_20251103` (old/all_views) est le **178** dans le nouveau jeu = code photo de MIA_10701,
    donc bien la paire 107.
  - Table de correspondance C1R1 : `metadata/c1r1_photocodes.csv` (417→101, 349/345→102, 111→103,
    207/104→104, 402→106, 178→107, 239→109, 188→110 ; pas de nid pour 105 et 108).
    `349_left_20251023` est daté du lendemain de sa séance (22/10) → rattaché avec ±1 jour de tolérance.
  - **Structure obtenue** (`outputs/inventory.csv`, colonnes `pair`, `cohort`, `round`, `session`,
    `nest_id` = MIA_<paire><ronde>, `use`) : **3 cohortes × 2 rondes, 25 paires, 49 nids, 175 séances,
    173 vues du dessus utilisables**. Cohorte 1 : ronde 1 oct.–nov. 2025, ronde 2 avr.–juin 2026 ;
    cohorte 2 : févr.–mars puis août–oct. 2026 ; cohorte 3 : avr.–juin puis août–oct. 2026. Une ronde =
    séances d'une paire espacées de ≤ 45 jours. Nids par cohorte × ronde : 8/7, 9/9, 8/8.
  - Cohérent avec le calendrier : pas de rouge en C1R1 (« Not provided to C1R1 », et les photos n'ont
    qu'orange / jaune) ; **noir à la place du rouge en C2R1** (cohorte 2, févr.–mars).
  - 4 doubles prises « (2) » (207 top 05/09, 309 top 12/08, 305 left 22/09, 308 right 10/05).
    **Explication (Renaud)** : quand le nid n'est pas un dôme, la vue d'entrée est prise dans l'axe d'une
    autre vue (dessus, ou côté) → rangée sous ce nom avec « (2) ». Règle : on garde la photo sans « (2) »
    (vue standard), la « (2) » est la vue d'entrée et est écartée comme les autres vues d'entrée.
  - 7 photos du dessus portent une rotation EXIF (6 ou 8) alors que les pixels sont déjà dans le bon sens
    (planche contact) : notre lecture ignore l'EXIF, c'est bien ; une visionneuse les affichera tournées.
  - 36 photos sont déjà dans `old/` (dont les 8 + 5 vues dessus déjà annotées).
- Constats visuels (planches contact des 178 vues du dessus) :
  - **Ficelle noire** à la place du rouge pour une série de 2xx en févr.–mars 2026 (201, 202, 203, 206,
    208, 209 : « Black instead of Red for C2R1 » dans le calendrier). **Nouvelle couleur**, et la plus
    difficile : elle se confond avec la classe trou / ombre.
  - **Deux rouges** : rouge framboise (102, 104, 107, 302–310 au printemps) et rouge foncé bordeaux
    (202, 203, 206, 310 en août) — même classe, mais l'entraînement actuel n'a vu que le premier.
  - **Matériau doré / pailleté** dans la coupe sur les séances tardives (202 30/09, 206 23/09, 303 06/10,
    309 29/05, 310 11/06…) ; carton, œufs ; **étiquettes roses numérotées** sur les nids d'oct.–nov. 2025
    (risque d'être classées rouge).
  - **Focale de 18 à 36 mm** : l'échelle varie d'une photo à l'autre (mire indispensable). Les plus grands
    angles ont un **vignettage noir circulaire** (106 15/04, 107 13/04, 201 10/02, 208 05/03, 309 12/08…).
  - Mire le plus souvent à gauche, parfois à droite (103 15/04, 206 24/08, 302 12/05, 309 29/05), parfois
    en partie cachée par la ficelle (204 févr.). Pas de grille verte en vue de dessus : la référence fixe
    serait le **cadre de la boîte** (planche de bois en haut, bords de la boîte).
  - Les nids d'oct.–nov. 2025 (111, 178, 188, 239, 345, 349, 402, 417, 207) n'ont qu'orange et/ou jaune :
    2–3 séances, construction précoce.

### Plan — vues du dessus (2026-11-20)
- **T0 — Métadonnées** (avant toute mesure).
  T0.1 ✅ Doublons et codes photo réglés ; paire, cohorte, ronde, nid, séance dans `outputs/inventory.csv`.
  T0.2 Reste : couleurs déjà fournies à la date de chaque photo et date du 1er œuf (calendrier d'apport
  par nid) → à obtenir de Marie ; c'est la variable « temps » de toute l'étude.
- **T1 — Échelle et repère** sur les 178 vues : mire (S1.1, déjà écrit, à tester sur mire à droite, cachée,
  vignettage) ; repère de position = cadre de la boîte (pour centre / périphérie et pour superposer les
  séances successives d'un même nid).
- **T2 — Segmentation, à refaire pour ce jeu** (Renaud annote) :
  T2.1 Classes : blanc, jaune, orange, rouge (framboise + bordeaux), **noir** (nouveau), trou / ombre, fond,
  autre (boîte, carton, mire, œufs, doré, étiquettes, vignettage).
  T2.2 Choisir ~15–20 images d'entraînement qui couvrent cohortes, couleurs, focales, vignettage ; les
  **séparer par nid** des images de validation (pas de fuite).
  T2.3 Outil d'annotation par rectangles (page HTML : tracer, choisir la classe, exporter le JSON au format
  `pixel_rects.json`) pour que Renaud annote sans coordonnées à la main.
  T2.4 Ré-entraîner (v6 comme point de départ : filtres de teinte corrigés), en ajoutant la classe noire ;
  traiter noir vs ombre (forme de brin / continuité S2.4).
- **T3 — Validation indépendante** : nouveaux points tirés sur les images de validation seulement
  (outil S2.3 existant), annotés par Renaud — et Marie sur un sous-ensemble pour l'accord entre annotateurs.
  Critère pour passer à la suite : couleur juste ≥ 0,9, ficelle manquée ≤ 10 %, noir ≥ 0,8.
- **T4 — Métriques sur les 178 vues** : composition (S3), indice de mélange en mm (S4), centre / périphérie
  par couleur (S5, vue du dessus), en fonction du temps (séance, couleurs fournies).
- **T5 — Suivi dans le temps (nouveau, propre au dessus)** : superposer les photos successives d'un même
  épisode (recalage sur le cadre de la boîte) → où la couleur nouvellement fournie apparaît (centre, bords,
  dessus de l'ancienne), et ce qu'elle recouvre. C'est la mesure la plus directe de la stratification.
- **T6 — Analyses (S8)** : modèles mixtes (nid / couple en effet aléatoire), cohorte, épisode 1 vs 2.
- Ensuite seulement : vues gauche / droite (hauteur, grille verte), puis S6–S7 (brins, dessus / dessous).

### 2026-11-20 — T1 : échelle des vues du dessus + T2.3 : page d'annotation
- `detect_scale.py` sur les 173 vues du dessus utilisables → `outputs/top/scale_top.csv`, planche
  `outputs/top/scale_check_top.jpg` (< 1 s par image). Mire trouvée sur 154, absente sur 19.
- Contrôle de cohérence : px/mm divisé par la focale EXIF ≈ constant (médiane 0,215, ±4 % entre quartiles ;
  la distance appareil–nid varie aussi, donc repère grossier seulement).
- Échecs vus à l'œil : **109 du 20/04 : échelle divisée par 2** (3,30 au lieu de ≈ 6,8 px/mm d'après les
  graduations) ; **206 du 13/03 : cases trouvées sur de la ficelle noire** ; 204 du 03/03 : mire en partie
  couverte de ficelle blanche ; les 19 absentes : colonne de cases cachée par la ficelle ou carte retournée.
- Filtre qualité (`outputs/top/scale_top_gate.csv`) : ≥ 5 rangs, résidu < 0,02, écart à la focale < 12 %
  → **115 échelles automatiques fiables, 58 à mesurer à la main** (clic sur deux graduations de la règle
  0–80 mm, quelques secondes par image). Plus rapide et plus sûr que de rendre la détection robuste à la
  ficelle noire et aux mires cachées.
- **Page d'annotation** : `python scripts/make_annotation_page.py` → `entrainement_rectangles/index.html` (ouvrir dans
  le navigateur depuis le dossier du projet). Deux modes : rectangles par classe (blanc, jaune, orange,
  rouge, **noir**, fond, autre, trou / ombre ; zoom molette, déplacement clic droit ou espace) et mire
  (deux clics + écart en mm). Sauvegarde automatique dans le navigateur, export d'un seul JSON
  (`rects` au format `pixel_rects.json` + `ruler`). Listes : 19 images d'entraînement, 58 mires à mesurer,
  les 173 vues. Rendu vérifié (capture Firefox headless) ; les interactions souris n'ont pas pu être
  testées automatiquement.
- **Séparation entraînement / validation par paire** : entraînement pris seulement dans les paires 101, 104,
  107, 202, 204, 206, 303, 308, 310 (19 images choisies pour couvrir : les deux rouges, le noir, le blanc
  dominant, le doré, les étiquettes, le vignettage, les focales extrêmes). Les 16 autres paires (dont 201,
  203, 208, 209 avec du noir, 205 et 207 avec beaucoup de blanc) servent à la validation T3.
- À faire côté code après l'annotation : classe « noir » (id, filtre de couleur, palette) dans `pixels.py`,
  lecture du nouveau format JSON par `pixel_classify.py`, échelles manuelles fusionnées avec les automatiques.
- **Renommage des deux pages** pour ne pas les confondre (2026-11-20) :
  `validation/` → **`validation_points_v1/`** (page de validation à l'aveugle, points tirés au hasard, jeu
  annoté par Marie sur les 32 anciennes photos, segmentation v5b ; renommé avec `git mv`) et
  `annotation/` → **`entrainement_rectangles/`** (rectangles d'entraînement + mire). Titres des pages
  changés en conséquence ; une explication pour l'annotatrice ajoutée dans la page d'entraînement.
  Scripts mis à jour (`revalidate`, `compare_variants`, `eval_validation`, `make_report_figures` ;
  `make_validation --out` désormais obligatoire) ; vérifié : `eval_validation` redonne 0,80. Le futur jeu
  de points sur les vues du dessus ira dans `validation_points_top/`. Les mentions de `validation/` plus
  haut dans ce journal renvoient à `validation_points_v1/`.
- `.gitignore` complété (2026-11-20) : photos et `.zip`, `models/` (régénérables à l'identique depuis
  `annotations/` + code, graine fixe), `outputs/experiments/`, `outputs/seg*/`, `outputs/all_views/`,
  caches Python. `models/pixel_rf.joblib` et les `__pycache__` étaient suivis : retirés du suivi
  (`git rm --cached`, fichiers conservés sur le disque). Les petits CSV et figures de `outputs/` restent suivis.
- `git gc --prune=now` : `.git` 507 Mo → 81 Mo (les ~400 Mo étaient des copies orphelines laissées par un
  `git add .` des modèles et sorties, pas l'historique). Restent dans l'historique, déjà sur GitHub : deux
  anciennes versions de `models/pixel_rf.joblib` (29 + 43 Mo) et `docs/Li et al. 2020` (15 Mo). Les retirer
  demande une réécriture d'historique + push forcé + re-clone par Marie → **non fait**, en attente de décision.
- Décision de Renaud : on garde le plan tel quel — **Marie annote les 19 images d'entraînement et les
  58 mires** dans `entrainement_rectangles/`, puis ré-entraînement (avec la classe noire) et validation T3.
  (Alternative écartée : passer d'abord v6 sur les 173 vues et n'annoter que les échecs.)
- 2026-11-23 — La page ne marchait pas chez Marie : elle chargeait les photos depuis `../05_Top/`, dossier
  exclu de git (volontairement, 139 Mo). Ajout d'un bouton **« Choisir le dossier des photos… »** : l'annotatrice
  désigne son dossier `05_Top` où qu'il soit, la page lit les photos localement (rien n'est envoyé nulle part) ;
  sans choix, elle tente toujours `../05_Top/`. Le dossier est à re-choisir à chaque ouverture de la page
  (limite des navigateurs) ; les annotations, elles, restent gardées. Rendu vérifié en local ; le sélecteur
  de dossier n'a pas pu être testé automatiquement.

### 2026-11-23 : Annotations par MD avec protocole "rectangles"
- annotations des 19 images d'entrainement par l'annotarice - fait sur le dossier `../05_Top_old/` (qui contient les noms originaux des photographies - le dossier `../05_Top/` contient les photos renommées de façon a ce que tous les noms contiennent le nom des paires/couples d'oiseaux)
- annotations des images de mires - mais **problème avec la case "écart"** qui reste identique pour toutes les mires (quand la valeur est changée, elle change pour toutes les images), si bien que seule les images avec mires entières (écart = 8cm) ont pu être traitées pour le moment. 
- import des nouvelles annotations dans le dossier "annotations"

### 2026-11-25 — Vérification des annotations de Marie (`annotations/annotations_top_MD_20261123.json`)
- Branche `origin/marie` fusionnée dans `master` (avance rapide, sans conflit).
- **Rectangles** : 581 sur les 19 images d'entraînement, toutes les classes (autre 131, rouge 89,
  orange 77, jaune 77, blanc 74, trou / ombre 55, fond 51, **noir 27**) ; toutes les photos retrouvées.
  Contrôle visuel (202 23/02, 204 03/03, 310 23/08, 308 17/05) : bien placés, sur les brins, y compris à
  l'ombre et sur les bouts fins — exactement les cas difficiles qui manquaient.
- **Mais très petits** : côté médian 11 px en pleine résolution (≈ 1,6 mm, une demi-largeur de brin) → peu de
  pixels d'entraînement : 500 à 3 000 par couleur à demi-résolution (noir : 523), contre 100 000–150 000 dans
  l'ancien jeu (rectangles larges filtrés par couleur). Des pixels très purs et variés, mais peu nombreux.
  Pistes : (1) ajouter les anciens rectangles (mêmes ficelles orange / jaune / blanc / rouge) ;
  (2) demander quelques rectangles **plus grands** le long des brins, surtout pour le noir (3 images seulement).
- La classe « trou / ombre » est annotée (55 rectangles) alors que le modèle la définit jusqu'ici par une
  règle de luminance : à décider à l'entraînement (l'apprendre comme une classe, ou s'en servir pour
  régler le seuil).
- **Mires** : 63 mesurées (49 des 58 demandées + 14 déjà automatiques). Sur ces 14, accord médian manuel /
  automatique **0,2 %** → les deux méthodes se valident. Deux mires enregistrées avec un écart de 70 mm
  (202 30/09, 206 06/03) sont fausses d'exactement 80/70 = +14 % : elles ont été cliquées de 0 à 80. C'est le
  **bug signalé par Marie** : le champ « écart » gardait la dernière valeur tapée d'une photo à l'autre (il
  n'était pas commun à toutes les photos, mais l'affichage le faisait croire, et un clic enregistrait la
  valeur affichée). **Corrigé** : chaque photo affiche son propre écart, 80 mm par défaut. Les deux mesures
  seront corrigées à 80 mm à l'import.
- Il reste **9 mires** (règle en partie cachée) ; la liste « mire à mesurer » de la page ne contient plus
  qu'elles (les mires déjà exportées dans `annotations/annotations_top_*.json` sont retirées). Consigne
  ajoutée : cliquer deux graduations visibles et taper leur écart.
- Dossier renommé de Marie (`05_Top` avec les noms de paires) : ses annotations sont indexées par les
  **noms d'origine** (`05_Top/104_top_20251107.JPG`…), cohérents avec l'inventaire (qui fait déjà la
  correspondance code photo → paire). Si on passe au dossier renommé, il faudra une table ancien → nouveau nom.

### 2026-11-25 — T2.4 : premiers entraînements sur les vues du dessus (v7)
- Code : classe **noir** (id 8, filtre sombre et neutre : chroma < 10, L < 40 ; affichée en bleu), classe
  **trou / ombre apprise** par l'étage A là où elle est annotée (la règle de luminance reste, seuil 0,45),
  plusieurs fichiers d'annotations avec plafond de pixels par fichier (`--ann a.json b.json --ann-cap 20000 1000`),
  format d'export de la page accepté, option **`--balance`** (chaque classe annotée pèse autant, quel que soit
  son nombre de pixels). Échelles : `scripts/merge_scale.py` → `outputs/top/scale_top_final.csv` (164 / 173 :
  63 manuelles, 101 automatiques ; 9 mires restantes ; les deux écarts 70 mm de Marie corrigés à 80 mm →
  accord manuel / auto 0,2 % médian, 0,8 % max).
- Mesures (367 anciens points, `compare_variants.py` ; pas de noir dans ces images) :

  | modèle | données | exact. | manqués | rappel blanc | précision orange | faux « ficelle » | corrigés / cassés vs v5b |
  |---|---|---|---|---|---|---|---|
  | v6 | anciens rectangles | 0,84 | 0,13 | 0,65 | 0,77 | 0,05 | 10 / 3 |
  | v7a | Marie seule | 0,82 | 0,16 | 0,47 | 0,90 | 0,15 | 22 / 18 |
  | v7b | Marie + anciens (1 000 px) | 0,87 | 0,14 | 0,47 | 0,85 | 0,04 | 21 / 8 (p = 0,02) |
  | v7c | v7b + `--balance` | **0,89** [0,84–0,93] | **0,03** | 0,76 | 0,88 | 0,10 | 25 / 8 (p = 0,004) |
  | v7d | v7c + 2 rectangles noirs larges | 0,86 | 0,01 | 0,82 | 0,88 | 0,16 | 24 / 13 |

  v7a n'a jamais vu ces images : test tout à fait indépendant ; couleur juste 0,99 quand les deux voient une
  couleur. Les autres incluent les anciens rectangles (points hors rectangles, mais mêmes images).
- **Noir** (contrôle visuel sur des paires de validation 201, 203, 208, 209) : v7b / v7c classent les nids
  noirs denses en « autre » — les 27 petits rectangles noirs de Marie sont tous sur des brins isolés sur fond
  clair (3 images, 508 px) : le modèle n'a jamais vu de masse noire. Test : 2 rectangles larges filtrés
  « noir » ajoutés par Claude sur 202 03/03 et 206 13/03 (paires d'entraînement, fichier à part
  `annotations/black_extra_claude.json`) → v7d reconnaît les nids noirs (209 19/02 : ficelle 7 % → 66 %).
  **Diagnostic confirmé** : il manque des exemples de noir dense. Mais v7d prédit du noir sur 16 des anciens
  points (images sans ficelle noire : trous sombres, intérieur de boîte) et les bouts noirs isolés en bordure
  restent « autre » ; le jaune recule sur 201.
- Suite proposée : demander à Marie des rectangles **noir** plus grands, sur du noir dense et sur des bouts
  isolés, et des rectangles **trou / ombre** dans des nids sans noir, sur 202 03/03, 206 13/03, 206 28/02
  (paires d'entraînement), puis réentraîner avec `--balance`. Pas encore de modèle adopté pour les vues du dessus.
- Question de Renaud sur le 209 : deux nids différents de la même paire (ronde 1 en févr. : jaune / orange le
  14/02, entièrement recouvert de noir le 19/02, œufs le 26/02 ; ronde 2 en août–sept. : blanc / jaune / orange,
  puis bordeaux par-dessus). Conséquence pour l'analyse : quand une couleur recouvre tout, la vue du dessus ne
  voit que la dernière couche ; c'est justement le signal de stratification (T5), mais les couleurs cachées
  ne se mesurent qu'avec les vues de côté.
- Renaud fait lui-même le complément (3 images) : liste « complément noir / ombre » ajoutée en tête de la page
  (`BLACK_EXTRA` dans `make_annotation_page.py`). Son export sera un fichier séparé
  (`annotations/annotations_top_RH_<date>.json`) ; rien n'écrase les rectangles de Marie (stockés dans son
  fichier, et chaque navigateur garde ses propres annotations). Les 2 rectangles de test de Claude
  (`black_extra_claude.json`) ne seront plus utilisés une fois ce complément fait.
- Complément de Renaud reçu : `annotations/annotations_top_RH_20261125_additionnal_black_and_hole.json`,
  30 rectangles (202 03/03 : 5 noir + 2 orange ; 206 13/03 : 2 noir + 3 ombre ; 206 28/02 : 8 jaune + 4 ombre ;
  plus 6 orange sur 104 07/11, distincts des 9 de Marie). Entraînement **v7e** = Marie + anciens (1 000 px) +
  Renaud (plafonds 20000 1000 20000), `--balance`, teinte 58/62, ombre 0,45, sans `black_extra_claude.json`
  (script `outputs/experiments/run_v7e.sh`, journal `run_v7e.log`).
- Résultats v7e (367 anciens points) : exact. 0,86 [0,81–0,91], ficelle 0,90, couleurs 0,94, manqués 0,03,
  rappel blanc 0,82, précision orange 0,94, faux « ficelle » 0,19 (v7c 0,10), 28 corrigés / 16 cassés vs v5b
  (p = 0,10). **Aucun point prédit noir** (v7d : 16) → les rectangles « ombre » de Renaud ont réglé les faux
  noirs dans les trous. Contrôle visuel (paires de validation) : noir dense reconnu (209 19/02 : noir 98 %
  de la ficelle, 203 24/02 : 99 %), jaune mieux gardé que v7d sur 201 17/02, blanc / noir séparés sur 208,
  205 (blanc) et 309 (rouge / jaune) corrects. Restent : bouts noirs fins en bordure encore « autre », et
  plus de faux « ficelle » que v7c (prédit ficelle là où les points disent fond / autre / ombre).
  Image : scratchpad `v7e_heldout.jpg` (v7d : `v7d_heldout.jpg`).
- **Décision (Renaud) : v7e adopté** pour les vues du dessus → `models/pixel_rf.joblib` (copie
  `models/pixel_rf_v7e.joblib` ; l'ancien défaut v6 reste dans `models/pixel_rf_v6.joblib`). Affinage des bouts
  noirs en bordure reporté.
- 9 dernières mires mesurées par Renaud (`annotations/annotations_top_RH_20261125_mire.json`, 40 mm sauf
  308 10/05 : 10 mm) → `merge_scale.py` : **173 / 173 vues ont une échelle** (72 manuelles, 101 auto).
  Contrôle : les 7 valeurs auto rejetées par le contrôle de qualité étaient à ≤ 1,5 % de la mesure manuelle
  (le contrôle était prudent) ; px/mm ÷ focale cohérent (0,19–0,25, médiane 0,215).
  Attention : ce fichier contient aussi les 30 rectangles du complément noir / ombre (la page garde tout) ;
  ne pas le passer en plus de `..._additionnal_black_and_hole.json` à l'entraînement (doublons).
- T4 : `scripts/top_metrics.py` (segmentation v7e + nettoyage S2.4 avec largeur de brin = 3,2 mm × échelle,
  cartes gardées dans `outputs/top/seg_v7e/`, métriques → `outputs/top/metrics_v7e.csv` : composition,
  surfaces en cm², mélange à 3–48 mm, contacts observés / attendus, centralité ; noir au rang du rouge).
  Lancé avec v7e, **arrêté à 56 / 173** : Renaud a repéré une erreur dans ses rectangles.
- Correction de Renaud : `annotations/annotations_top_RH_20261125_mire_and_extra_black.json` (9 mires + rectangles
  corrigés ; les deux anciens fichiers RH déplacés dans `annotations/old/`). Changements : sur 206 28/02 un
  rectangle ombre et un jaune étaient inversés ; sur 206 13/03 les 2 grands rectangles noirs remplacés par 5 plus
  petits sur les brins et 3 petits « ombre » ; sur 202 03/03 1 noir déplacé, +1 noir, +3 orange.
  → réentraînement **v7f** (mêmes réglages que v7e, `outputs/experiments/run_v7f.sh`), puis T4 complet avec v7f
  (`outputs/top/seg_v7f/`, `outputs/top/metrics_v7f.csv`). Les 56 cartes v7e partielles restent dans `seg_v7e/`.
- v7f (367 anciens points) : exact. 0,86 [0,80–0,90], manqués 0,05, rappel blanc 0,82, précision orange 0,94,
  faux « ficelle » 0,16 (v7e 0,19), 0 point noir ; photos de contrôle quasi identiques à v7e (noir 98–99 % sur
  209 / 203). **Adopté** → `models/pixel_rf.joblib` (copie `pixel_rf_v7f.joblib`, v7e gardé).
- **T4 terminé** : 173 / 173 photos, surface pour toutes. Contrôles de cohérence avec le protocole :
  noir > 2 % hors cohorte 2 ronde 1 sur 2 photos seulement (107 13/04 : 4 %, 308 17/05 : 3 % — les deux à
  vignettage) ; rouge > 2 % en ronde 1 des cohortes 1–2 sur 1 seule photo (349 22/10 : 3 %). Composition
  moyenne : C1R1 orange 0,68 / jaune 0,29 ; C2R1 noir 0,33 ; rouge ≈ 0,3 en ronde 2, 0,53 en C3R1.
  Ficelle = 39–89 % du nid (ficelle + autre + ombre), surface visible 126–651 cm² (médiane 317).
  Mélange : défini sur 138 photos (pas calculé quand une couleur domine à > ~97 %).
- Figures T4 : `scripts/plot_top_metrics.py` → `outputs/top/fig_composition_v7f.png` (un panneau par nid,
  une barre par séance) et `fig_trends_v7f.png` (surface, mélange à 12 mm, rho ordre ↔ centre par séance).
  Constats :
  1. ~~L'ordre des couleurs n'est pas le même pour tous les nids~~ — **conclusion fausse de Claude, corrigée
     par Renaud (2026-11-26)** : l'ordre est toujours orange → jaune → blanc ; après le 1er œuf : rien en C1R1,
     noir en C2R1, rouge pour toutes les autres rondes. Un nid « jaune » ou « blanc » dès la 1re photo du
     dessus (103, 109, 110 en C1R1 ; 204, 205, 208 en C2R1) avait simplement déjà recouvert l'orange (et le
     jaune) au moment de la photo : la vue du dessus ne montre que la dernière couche. `PROVISION_ORDER`
     (noir au rang du rouge) est donc correct et `strat_rho` interprétable. Les dates de remise par nid
     (T0.2) restent utiles pour T5 (savoir quelle couleur était disponible à chaque séance), pas bloquantes.
  2. La vue du dessus est dominée par la **dernière couleur donnée** (ex. 203 : orange → 100 % noir en 5 jours ;
     rouge qui passe à 80–95 % en C3R1) : recouvrement rapide, signal de stratification direct.
  3. Surface visible stable ou en baisse au fil des séances (C1R2, C2R1) : le nid se tasse / la coupe se
     creuse ; à vérifier sur les photos.
  4. Indice de mélange peu fiable quand une couleur domine (> ~90 %) : C1R1 quasi monochrome, valeurs > 1
     sur des couleurs très minoritaires → à ne calculer que si la 2e couleur fait ≥ 10 % (à faire).
     **Fait** (`MIN_SECOND = 0.10` dans `top_metrics.py`, colonne `second_share`) : mélange défini sur 104
     photos (C1R1 : 3 seulement), max 0,97 ; médiane à 12 mm 0,53 (C2R1) à 0,77 (C2R2). Figures régénérées.
- 2026-11-26 — Test de v7f sur 4 vues de côté (203 left 26/02, 201 right 17/02, 208 left 09/03, 309 right
  15/04 ; image scratchpad `side_v7f.jpg`), pour évaluer l'intérêt des vues latérales (345 photos, 49 nids) :
  couleurs justes là où la ficelle est trouvée ; 208 montre la stratification attendue de bas en haut
  (orange → jaune → blanc → noir). Mais l'étage A n'a jamais vu de vue de côté : ficelle noire classée
  « ombre » (203), ficelle dans la boîte en carton classée « autre », boîte / grille / fond du studio pas
  toujours séparés. → vues de côté = annotation de rectangles sur ~10–15 vues latérales + réentraînement.
  Décision reportée après le pipeline dessus complet.
- 2026-11-26 — Bilan « vues du dessus seules » vs questions de Marie (`docs/Quantifying_Interlacing_Weaving.pdf`) :
  approche 1 (indice de mélange par cellules) → `window_entropy_curve` existe mais **pas encore dans
  `top_metrics.py`** ; approche 2 (voisinage brin à brin) → approximée au pixel (contacts observés / attendus),
  brins individuels (S6) non faits ; approche 3 (autocorrélation / semi-variogramme) → c'est l'indice de mélange
  multi-distance 3–48 mm ; 4 (texture, Bailey 2015) et 5 (dessus / dessous, Li 2020) non faits.
  Résultats provisoires (v7f, sans validation dessus) : rho ordre ↔ centre médian +0,11 à +0,18 (54–74 % des
  photos > 0) sauf C1R1 (−0,10) ; centralité médiane orange 0,28, blanc 0,28, jaune 0,40, rouge 0,39,
  noir 0,42 ; dernière séance : couleur dominante 75 % (médiane), rouge + noir 48 % ; mélange à 12 mm par
  nid 0,23–0,91 (37 nids). Rappel du calendrier (`docs/string_provisionning_schedule.pdf`) : jaune et blanc
  peuvent être sautés.
- 2026-11-26 — Précision de Renaud : dès le 1er œuf on passe directement à la dernière couleur (rouge ou noir),
  d'où des nids sans blanc ni jaune. Vitesse de recouvrement abandonnée → les dates de remise par nid ne
  sont plus demandées.
- **Correction de l'indice de mélange** (`metrics.contact_matrix`) : les décalages diagonaux valaient
  lag en x et en y (distance × 1,41) ; maintenant même distance euclidienne dans les 4 directions.
  Tests synthétiques : deux moitiés 0,03 à 1 w, hasard 1,00 à toutes les distances. Change aussi les
  anciens résultats (`metrics.py` sur l'ancien jeu) si on les recalcule.
- **Approche 1 de Marie ajoutée** à `top_metrics.py` : entropie moyenne des couleurs dans des cellules carrées
  de 6 à 102 mm (2–32 largeurs de brin, cellules au moins à moitié ficelle), divisée par l'entropie du nid
  entier (`cellH_<n>mm` ; 0 = chaque cellule d'une seule couleur, 1 = chaque cellule aussi mélangée que le nid).
  Tests : deux moitiés 0 → 0,95 selon la taille, hasard 1,00.
  Recalcul (`metrics_v7f.csv`, figures régénérées) : médianes mélange 0,30 / 0,51 / 0,66 / 0,80 / 0,91 à
  3 / 6 / 12 / 24 / 48 mm (un peu plus bas qu'avant : les diagonales étaient trop longues) ; cellules 0,29 →
  0,88 de 6 à 102 mm. **Les deux approches classent les photos pareil** (Spearman 0,93–0,95 entre mélange à
  12 mm et cellules de 13 / 26 mm). Exemple 107 13/04 vs 302 18/08 : 0,38 vs 0,86 (mélange 12 mm),
  0,27 vs 0,65 (cellules 13 mm).
- **T3 lancé : page de validation à l'aveugle des vues du dessus** → `validation_points_top/index.html`
  (370 points : 210 uniformes = 2 par photo, 160 stratifiés = 20 par classe prédite, v7f, sur les 105 vues
  du dessus des 16 paires de validation, hors rectangles d'entraînement). Page : bouton **noir** ajouté
  (touches 1 à 9), clé de stockage propre à la page (`--key nestweave_validation_top_v1`) pour ne pas
  recharger les réponses de l'ancienne. `make_validation.py` : `--only`, `--rects` multiples, `--key` ;
  `eval_validation.py` et `compare_variants.py` comptent le noir comme couleur.
  Évaluation une fois annoté : `cd scripts && python eval_validation.py ../annotations/<fichier>.json
  --dir ../validation_points_top`.
- Commit `37d154c` (v7f, échelles, T4, page de validation dessus ; `outputs/top/seg_*/` et logs ignorés).
- **T6 — premiers modèles statistiques** (`scripts/stats_top.py` → `outputs/top/stats_v7f.txt`,
  `fig_pair_signature_v7f.png`). `statsmodels` 0.15 ajouté à l'environnement (`environment.yml`).
  Modèles mixtes (MixedLM, REML) : couple en effet aléatoire + nid dans le couple ; effets fixes cohorte,
  ronde, jours depuis la 1re photo du nid. PROVISOIRE (pas de validation dessus).
  - Q1 mélange à 12 mm (104 photos, 37 nids) : pas d'effet cohorte ; **2e nid un peu plus mélangé**
    (+0,08 ; p = 0,06 ; cellules 26 mm : +0,08, p = 0,03) ; pas d'effet des jours. Variance : 0 % couple,
    44 % nid, 56 % photo.
  - Q2 signature du couple : **aucune** (R ≈ −0,2 à −0,3, IC jusqu'à +0,4 ; Spearman 1er vs 2e nid ≈ 0) ;
    seulement 13 couples avec le mélange défini sur leurs deux nids → peu de puissance.
  - Q3 couleurs récentes vers le centre : **oui, faiblement** — rho moyen +0,11 [0,07 ; 0,15], 61 % des
    photos > 0 ; plus fort en cohortes 2 et 3 qu'en cohorte 1 (+0,10 à +0,11, p ≈ 0,02). Centralité médiane :
    orange 0,28, blanc 0,28, jaune 0,40, rouge 0,39, noir 0,42 → le blanc ne suit pas l'ordre (à comprendre).
  - Limites : p-values non corrigées pour tests multiples ; pixels ≠ brins ; mélange non défini quand une
    couleur couvre tout (C1R1 : 3 photos) ; la ronde 2 a du rouge, la C1R1 rien après l'œuf (confusion
    possible ronde ↔ couleurs disponibles).
- **T3 — Validation à l'aveugle des vues du dessus (Renaud, `annotations/validations_top_RH_20261126.json`)** :
  370 points sur les 16 paires de validation, 34 « incertain » (9 % ; 21 uniformes, 13 stratifiés ; surtout
  sur des points prédits rouge 9, blanc 6, orange 5). Modèle v7f :
  exactitude **0,92 [0,87–0,95]** (189 uniformes) ; sur la ficelle 0,93 ; **couleur juste 0,99** quand vérité et
  prédiction sont des couleurs ; ficelle perdue en ombre 2 %. Précision : jaune 0,93, orange 0,89, rouge 0,91,
  **noir 0,97**, blanc 0,82 (5 erreurs = fond / boîte pris pour du blanc), ombre 0,54. Rappel : jaune 1,00,
  orange 0,97, rouge 0,92, blanc 0,90, noir 0,90 (10 points), ombre 0,50.
  **Critères de T3 remplis** (couleur ≥ 0,9, ficelle manquée ≤ 10 %, noir ≥ 0,8). Borne pessimiste si tous
  les « incertain » uniformes étaient des erreurs : 173 / 210 = 0,82. Pas d'accord inter-annotateurs (un seul
  annotateur) ; la segmentation n'est pas modifiée → les résultats T4 / T6 deviennent validés (v7f).
- **Compte rendu en anglais pour Marie** : `reports/top_views/main.tex` → `main.pdf` (9 pages : données,
  pipeline, versions du modèle, validation, métriques avec formule et exemple 107 / 302, résultats, limites,
  suite). Tous les chiffres, tableaux et figures viennent de `scripts/make_top_report.py` (macros
  `tables/values.tex`), rien n'est recopié à la main. Au passage : `plot_top_metrics.py` et `stats_top.py`
  acceptent des légendes anglaises (`lang="en"`), figure « signature du couple » sortie dans
  `stats_top.pair_figure`. Correction : borne pessimiste de la validation = 173 / 210 = 0,82 (et non 0,83) ;
  mélange médian par nid après correction des diagonales = 0,21–0,87.
  Détail LaTeX : les tableaux inclus dans un `tabular` utilisent `\csname @@input\endcsname` (un `\input`
  ordinaire provoque « Misplaced \noalign » avant `\bottomrule`).
- Modèle « ronde » refait à toutes les échelles (question de Renaud) : effet ronde 2 toujours positif mais
  faible et rarement significatif — mélange 3 / 6 / 12 / 24 / 48 mm : +0,05 / +0,07 / +0,08 / +0,07 / +0,05
  (p 0,06–0,15) ; cellules 6 / 13 / 26 / 51 / 102 mm : +0,04 / +0,06 / +0,08 / +0,06 / +0,02 (p 0,035–0,54).
  Variance couple ≈ 0 % à toutes les échelles ; part « photo » croît avec la distance (45 → 73 %).
  → à présenter comme une **tendance**, pas un résultat (le rapport cite seulement 12 / 26 mm : à nuancer).
  **Fait** : rapport mis à jour (10 pages) — tableau de l'effet ronde à toutes les échelles
  (`tables/round_scales.tex`, généré par `make_top_report.py`), résumé et résultats reformulés en
  « tendance », explication des trois niveaux de variance (couple / nid / photo), phrase du protocole
  après le 1er œuf réécrite.

### 2026-11-26 — Plan T7 : stratification avec les vues de côté
- État des lieux : 345 vues de côté utilisables (172 gauche, 173 droite), 171 / 174 séances ont les deux côtés.
  `detect_scale.py` relancé sur toutes (`outputs/side/scale_side.csv`, planche `scale_check_side.jpg`) :
  mire trouvée sur 333 / 345 (278 passent le contrôle de qualité du dessus) ; haut de la grille (zéro de
  hauteur) trouvé directement sur 252, par médiane de séance sur 87, pas du tout sur 6 ; écart barre haute –
  barre basse sur les 138 vues à 3 barres : médiane 41,0 mm (comme sur l'ancien jeu : 40,5–43,1), mais
  extrêmes 19 et 49 mm → contrôle de qualité à ajouter.
- Plan (détail dans la réponse à Renaud) : T7.1 échelle + zéro (contrôle, mesure manuelle des échecs dans
  la page) ; T7.2 rectangles sur ~12 vues de côté des paires d'entraînement ; T7.3 réentraînement (modèle
  commun dessus + côté, vérifié sur les points de validation du dessus déjà annotés) ; T7.4 validation à
  l'aveugle sur les côtés des paires de validation ; T7.5 métriques de hauteur (mm au-dessus de la grille,
  profil de couleurs par tranche de hauteur, rho ordre ↔ hauteur, recouvrement entre couleurs) ; T7.6
  modèles mixtes (couple / nid / vue) et lien avec le dessus ; T7.7 rapport.
  5. Nids à une seule photo du dessus : 205 R1 (03/04), 205 R2 (12/10).

## Questions ouvertes

- Quantité de ficelle fournie par phase : identique pour chaque couleur ?
    - 150 ficelles de chaque couleur données aux oiseaux. Au moins 125 utilisés pour passer à la couleur suivante (ou ponte d'un oeuf). A noter les ficelles utilisées peuvent aussi tomber hors du nid par la suite, donc le total utilisé quand la couleur a été fournie n'est pas forcément équivalent á la quantité présente dans le nid a un moment T ultérieur. 
    - comment prendre en compte le nombre de couleurs dans le nid pour calculer un indice de mélange des couleurs ? idéalement il faudrait savoir combien de pièces de chaue couleur sont dans le nid au moment de la photo, MAIS ces données ne sont pas présentement accessibles. Donc le plus simple est de considérer le nombre de couleurs utilisées jusque là (indépendamment du nombre de piéces de chaque couleur)
- Numérotation des nids (107, 206, 308…) : couple, pièce, volière ?
    - les nids sont numérotés par paires/couples d'oiseaux. 2 nids construits par couple. 3 séries de photo par nid. (Seule la 1ére série de photos, sans matériaux rouges, sera analysée dans un 1er temps)
- Séries temporelles après le 1er œuf : elles couvrent la phase rouge, donc peut-être utiles pour voir où le rouge est ajouté.

### 2026-11-27 — Kit autonome « vues du dessus » pour Marie (hors GitHub)
- `kit/top_views/top_report.py` : une fonction `make_report(...)` + ligne de commande, sans le dépôt ni Claude (numpy, pandas, matplotlib, scipy, statsmodels). Refait toutes les figures (exemples, mélange vs échelle, composition, évolution, centralité, signature du couple) et tous les tableaux (groupes, chiffres clés, modèles mixtes, effet ronde à toutes les échelles, répétabilité, validation) en CSV + `index.html`.
- Options : `--cohorts`, `--rounds`, `--max-session N` (garde les photos 1..N), `--drop-sessions`, `--pairs`, `--exclude-pairs`, `--exclude-photos`, `--n-boot`.
- Données du kit : `metrics_top.csv` (= metrics_v7f), `validation_points.csv` (points + prédiction + réponse), 6 overlays d'exemple à mi-résolution. Zip 0,7 Mo : `kit/top_views_kit.zip`, fabriqué par `kit/build_top_kit.py`. `kit/` ajouté au .gitignore (envoyé à la main, jamais poussé).
- Vérifié : sur les 173 photos, coefficients, parts de variance, répétabilité et validation identiques au rapport.
- Corrections en route : (1) une seule ronde → chaque couple n'a qu'un nid, couple et nid confondus → modèle singulier ; on retire alors la composante nid (la part « couple » = couple + nid). (2) `--rounds 2` : l'optimiseur lbfgs échoue (« Singular matrix ») ; repli automatique bfgs → powell → nm (même solution).
- Premier aperçu `--max-session 3` (138 photos) : l'effet ronde 2 sur le mélange baisse (+0,07, p = 0,19) et l'effet « jours » devient significatif (+0,008/jour, p = 0,001) — le mélange augmente pendant la construction, la photo 4 pèse dans le résultat. À regarder.
- **Rangement du kit** (même jour) : un seul dossier `kit/top_views_kit/` (= ce que reçoit Marie, le zip se décompresse en `top_views_kit/`). L'ancien `kit/top_views/` est renommé ; la copie décompressée en double est supprimée, après avoir gardé le premier `results/` de Renaud dans `kit/old_results/results_first_run`. Le script de construction devient `kit/build_top_kit.py` ; la source du guide est dans `kit/guide/`.
- **Option `--time session`** : le temps entre dans les modèles comme un effet par numéro de photo (`C(session)`) au lieu d'une droite en jours.
- **GUIDE.pdf** (3 pages, en anglais, dans le kit) : pourquoi un modèle mixte, comment lire les coefficients (intercept = cohorte 1, ronde 1, jour 0 ; les autres lignes sont des écarts ; IC, p) et les parts de variance, l'effet du temps, des précautions.
- **Correction de mon interprétation d'hier.** Ce n'est pas le retrait de la photo 4 qui « révèle » l'effet du temps. Le mélange monte de la photo 1 à la photo 3 (+0,09 puis +0,11, p ≤ 0,002), puis plafonne à la photo 4 (+0,12, vers J41, après éclosion). Une droite sur J0–J41 écrase cette montée, d'où la pente faible avec toutes les photos.
- Avec `--time session` sur toutes les photos, l'effet ronde 2 est +0,07 [−0,02 ; +0,16], p = 0,14 : ça reste une tendance.
- **Le kit va sur GitHub** (demande de Renaud). Seuls restent ignorés le zip, les sous-produits LaTeX du guide et les dossiers `results*/` du kit. `kit/old_results/` est supprimé.
- **Minimum pour lancer le script :** `top_report.py` + `data/metrics_top.csv`.
  - `validation_points.csv` devient optionnel : sans lui, les tableaux de validation sont sautés.
  - `overlays/` est optionnel : sans eux, la figure d'exemples est sautée.
  - Testé dans un dossier qui ne contenait que ces deux fichiers.
- **Images d'exemple** (`kit/top_views_kit/data/overlays/`) : elles restent hors du dépôt (.gitignore) mais sont toujours dans le zip. `kit/guide/guide.pdf` est ignoré, car il fait doublon avec `GUIDE.pdf`. Commit du kit et de `outputs/side/scale_side.csv`. `scale_check_side.jpg` (7 Mo) n'est pas commité.
