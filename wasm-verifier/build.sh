#!/bin/bash
# Build WASM verifier and patch for browser compatibility

set -e

echo "Building WASM verifier..."
wasm-pack build --target web --out-dir pkg

echo "Patching for browser compatibility..."
node patch.js

echo "Copying to root directory..."
cp pkg/zkml_wasm_verifier*.{js,wasm,ts} ../

echo "✅ WASM verifier built and patched successfully!"
echo "📦 Output: pkg/ (copied to root)"
