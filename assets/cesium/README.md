# Cesium HQ Asset Setup

1. Copy `asset_manifest.example.json` to `asset_manifest.json`.
2. Fill in:
   - `ionToken` for Cesium World Terrain/Imagery and Ion assets.
   - `googleMapsApiKey` for Google Photorealistic 3D Tiles.
   - `tileset.ionAssetId` or `tileset.url` for photoreal 3D tiles (optional but recommended).
   - model URIs for drone, fire FX, buildings, park props, and responders.
3. Host your model files under `assets/models/...` or provide absolute HTTPS URLs.
4. Open `http://localhost:9090/cesium_viewer.html` and click **Apply Scene Config**.

Notes:
- With `strictMode: true`, missing required model URIs will block rendering to avoid low-quality fallbacks.
- The viewer auto-loads `/assets/cesium/asset_manifest.json` if present.
