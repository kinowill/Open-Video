# Open-Video

Chaîne de rendu vidéo locale sous Windows : rendu Blender/Cycles, export MP4 avec FFmpeg, reprise des images et audit des rendus. Les scènes, médias et captures ne sont pas fournis.

## Rendre une scène

Prépare une scène Blender avec une caméra active. Les chemins des exemples ci-dessous sont relatifs ; remplace-les par tes propres fichiers.

```powershell
.\open-video.ps1 -Action initialiser
.\open-video.ps1 -Action produire -Scene .\cache\scene.blend -Recette .\essais\pov\recette.json
```

`initialiser` installe les outils portables dans `outils/` à partir des versions et empreintes SHA-256 de `outils.lock.json`. Le pipeline de rendu vise Windows avec un GPU NVIDIA compatible OptiX. Les sorties et journaux restent par défaut dans `cache/rendus/`. Évite de partager ces journaux bruts : ils peuvent contenir des chemins locaux.

Une recette JSON indique notamment `resolution`, `fps` et `frames`. Pour ajouter une piste audio facultative, renseigne `audio.path`, relatif à la recette, et éventuellement `audio.offset_seconds`. Une recette avec une seule image produit un PNG.

- `produire -Images (1..3)` rend une sélection d'images. `produire -Reprendre` réutilise les images vérifiées lorsqu'on garde la même scène, les mêmes réglages et le même dossier `-Sortie`.
- `auditer` compare les profils de rendu sur une scène fournie avec `-Scene`.
- `preparer-studio` assemble un plan à partir d'une bibliothèque Blender fournie avec `-Scene` et choisit une collection avec `-Collection`. Pour activer le socle, règle `studio.plinth` dans la recette et fournis une bibliothèque de pierre avec `-Stone`.
- `materiau` génère une bibliothèque `stone.blend` dans son dossier de sortie. `preparer-pov` demande cette bibliothèque (`-Stone`) et un dossier de ressources (`-Assets`). Voir [la disposition attendue](docs/ASSETS.md).
- Les commandes optionnelles `initialiser-photo` et `initialiser-reconstruction` préparent les outils pour le traitement des photos et la reconstruction 3D. Elles téléchargent des dépendances externes.

Voir [le document de suivi](docs/DOCUMENT_MAITRE.md) pour le périmètre et les validations.
