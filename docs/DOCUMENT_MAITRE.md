# Open-Video — périmètre public

Le dépôt fournit le code du pipeline et des recettes génériques. Il est indépendant, sans historique antérieur, sans médias, bibliothèques Blender ni archives de production. Les entrées de rendu sont fournies par l'utilisateur ; les sorties ne sont pas versionnées.

La compatibilité visée est Windows, PowerShell, Blender/Cycles avec OptiX et FFmpeg. Les outils et leurs empreintes sont consignés dans `outils.lock.json`. Les modules de reconstruction demandent une installation optionnelle et des entrées locales.

Les contrôles de publication portent sur la syntaxe, les recettes, la cohérence des entrées du lanceur et l'absence de chemins personnels, d'identifiants connus et de fichiers binaires dans les fichiers publiés. Un rendu complet Blender/Cycles et l'installation de tous les outils restent à tester sur un poste compatible.
