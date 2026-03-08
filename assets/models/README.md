# Model Drop Folders

Put `.glb`/`.gltf` files in these folders:

- `assets/models/drone`
- `assets/models/fire`
- `assets/models/building`
- `assets/models/park`
- `assets/models/responder`

`serve.py` auto-selects the first model file in each folder when the corresponding `.env` variable is not set.

Priority order:

1. Explicit `.env` URI (e.g. `FIRE_MODEL_URI`)
2. Auto-discovered file from folder above
3. Built-in fallback (drone only)
