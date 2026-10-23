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

Défauts restants après S2.4 :
- bord ombré des brins jaunes classé orange : en partie seulement corrigé par S2.4 ; risque de gonfler les
  contacts hétéro-couleur (S4) ;
- le blanc à l'ombre au fond de la coupe est classé « support » (207 front) ; brin beige-gris devant la boîte
  en partie raté ;
- extrémités fines de brins rouges sur fond blanc classées « support » (206, bord gauche) ;
- halo « support » autour du nid ;
- mire et bois pas encore détectés automatiquement (pas encore d'échelle px/mm).

## Questions ouvertes

- Quantité de ficelle fournie par phase : identique pour chaque couleur ?
- Numérotation des nids (107, 206, 308…) : couple, pièce, volière ?
- Séries temporelles après le 1er œuf : elles couvrent la phase rouge, donc peut-être utiles pour voir où le rouge est ajouté.
