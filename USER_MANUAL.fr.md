# Manuel Utilisateur LTSpiceBatch

Ce manuel explique comment utiliser LTSpiceBatch, de la configuration initiale jusqu'aux images/vidéos finales.

## 1. Prérequis

Vous avez besoin de :

- LTSpice installé (requis).
- FFmpeg installe et accessible par chemin (inclus).
- Les dependances Python installees (laissez `uv` s'en charger).
- Un schéma LTSpice valide (`.asc`) avec les traces et paramètres a balayer.

## 2. Lancer l'application

### Windows

Commande :

```powershell
./run-sim.bat
```

### Linux

Commande :

```sh
./run-sim.sh
```

Cela ouvre l'interface graphique (`launcher.py`).

## 3. Configurer les paramètres globaux

Dans l'onglet Global Config :

1. Définir `input_file` vers votre fichier `.asc`.
2. Ajuster au besoin :
   - `ffmpeg_bin`
   - `ffmpeg_framerate`
   - resolution/DPI dans `image`
   - `parallel_sim`
   - `parallel_plot`
   - `runner_timeout`
   - `temp_folder`

Important :

- Si vous utilisez `temp_folder`, le dossier doit être accessible en écriture.

## 4. Créer des étapes de simulation

Dans l'onglet Steps, cliquer sur Add Step.

Champs requis :

- `name`
- `sim_command` (exemples : `.ac dec 100 20 20k`, `.tran {10/freq}`)
- `tracestoplot` (une ou plusieurs traces LTSpice)

Champs optionnels :

- `ffmpeg_framerate` par étape
- `fft` pour générer les graphes FFT en transient
- `fft_x_max`
- bornes d'axe (`y_min`, `y_max`, `mag_y_min`, `mag_y_max`, `phase_y_min`, `phase_y_max`)

## 5. Définir les parametres de balayage

Chaque paramètre peut être défini de trois façons :

1. Valeur unique
2. Liste de valeurs
3. Plage (`start`, `stop`, `step`)

Exemple :

```yaml
parameters:
  c2: 6.8n
  freq: [10, 73, 259, 657, "1k"]
  c1:
    start: 1e-12
    stop: 1e-10
    step: 2e-12
```

Les combinaisons sont générées par un produit cartésien.

## 6. Sauvegarder et exécuter

Dans l'onglet Save and Run :

1. Cliquer sur Save YAML (ou Save and Run directement).
2. Choisir le fichier `.yml` ou `.yaml`.
3. Suivre les logs dans la zone de sortie.
4. Utiliser Stop pour demander une interruption propre.

## 7. Mode ligne de commande

Exécution sans GUI :

```sh
uv run LTSpiceBatch.py -c path/to/config.yml
```

Options utiles :

- `--encode-only` : encoder uniquement depuis des images existantes.
- `--skip-encode` : faire simulation + graphes sans encoder la vidéo.
- `--keep-images` : conserver les images et les copier dans un dossier horodaté.
- `--reset` : forcer un re-run meme si des fichiers `.raw` existent.
- `--cleanup` : nettoyer le dossier temporaire.
- `--show-freq-domains` : afficher les bandes de frequence en AC.

## 8. Comprendre les sorties

Images générées (dossier temporaire) :

- `<step>-<param_set>.png`
- `fft_<step>-<param_set>.png` si FFT active

Vidéos générées (à côté du `.asc`) :

- `<input>.<step>.mp4`
- `<input>.<step>_fft_.mp4` si FFT encodée

Avec `--keep-images`, les images sont aussi copiees dans un dossier horodaté.

## 9. Visualiser avec l'outil diapo

### Windows

```powershell
./run-show.bat
```

### Linux

```sh
./run-show.sh
```

Puis :

1. Ouvrir un dossier contenant les PNG (et éventuellement `*_imglist.txt`, recommandé pour préserver l'ordre).
2. Naviguer avec les boutons ou la molette.
3. Utiliser les filtres bases sur les metadonnées JSON des images.

## 10. Dépannage

### FFmpeg introuvable

- Donner un chemin absolu dans `ffmpeg_bin`.
- Vérifier les droits d'exécution.

### Traces manquantes

- Verifier que chaque valeur de `tracestoplot` correspond exactement aux traces du `.raw` LTSpice.

### Simulations sautées

- Les fichiers `.raw` existants sont réutilisés sauf si `--reset` est active.

### Linux et compatibilité LTSpice

- Renseigner les champs Wine dans l'interface :
  - `wine_executable`
  - `wine_folder`

### Jobs longs ou lourds

- Réduire `parallel_sim` et `parallel_plot`.
- Augmenter `runner_timeout`.

## 11. Workflow recommandé

1. Partir du fichier exemple.
2. Valider une petite étape d'abord.
3. Elargir progressivement les balayages.
4. Activer FFT uniquement quand nécessaire.
5. Conserver les images pendant la phase de mise au point (si vous ne voulez que les vidéos).
