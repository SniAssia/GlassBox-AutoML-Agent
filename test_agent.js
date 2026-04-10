const fs = require('fs');
const path = require('path');
const { run_automl_bridge, initSync } = require('./pkg/glassbox_ironclaw_bridge.js');

async function main() {
    console.log("🤖 IronClaw Agent se réveille...");

    try {
        // 1. Charger manuellement le fichier .wasm pour Node.js
        const wasmPath = path.join(__dirname, 'pkg', 'glassbox_ironclaw_bridge_bg.wasm');
        const wasmBuffer = fs.readFileSync(wasmPath);
        
        // 2. Initialiser le module IronClaw
        initSync({ module: wasmBuffer });
        console.log("⚙️  Moteur Rust IronClaw initialisé.");

        // 3. Simuler les données (Base64 d'un petit CSV)
        const mockCsvB64 = "Y29sMSx0YXJnZXQKMSwwCjIsMQo=";
        
        console.log("📡 IronClaw envoie l'ordre au serveur Python...");
        
        // 4. Appel du Bridge Rust
        const result = await run_automl_bridge(mockCsvB64, "target", "classification");
        
        console.log("✅ IronClaw a reçu la réponse du Zoo de modèles :");
        console.log(JSON.stringify(result, null, 2));

    } catch (error) {
        console.error("❌ Erreur de l'Agent :", error);
    }
}

main();