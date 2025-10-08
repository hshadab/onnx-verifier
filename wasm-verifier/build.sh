#!/bin/bash
# Build WASM verifier and patch for browser compatibility

set -e

echo "Building WASM verifier..."
wasm-pack build --target web --out-dir pkg

echo "Patching for browser compatibility..."
# Patch externref table initialization to handle browsers without full support
sed -i 's/imports\.wbg\.__wbindgen_init_externref_table = function() {/imports.wbg.__wbindgen_init_externref_table = function() {\n        try {/' pkg/zkml_wasm_verifier.js
sed -i '/imports\.wbg\.__wbindgen_init_externref_table/,/};$/{s/;$/;\n        } catch (e) {\n            \/\/ Browser doesn'\''t support externref table.grow() - fallback to basic initialization\n            console.warn('\''[WASM] Browser compatibility: externref table initialization skipped:'\'', e.message);\n        }\n    };/}' pkg/zkml_wasm_verifier.js

echo "Copying to root directory..."
cp pkg/zkml_wasm_verifier*.{js,wasm,ts} ../

echo "✅ WASM verifier built and patched successfully!"
echo "📦 Output: pkg/ (copied to root)"
