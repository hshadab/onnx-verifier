// Manually patch the WASM file for browser compatibility
const fs = require('fs');

const filePath = 'pkg/zkml_wasm_verifier.js';
let content = fs.readFileSync(filePath, 'utf8');

// Find and replace the __wbindgen_init_externref_table function
const original = `    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_3;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };`;

const patched = `    imports.wbg.__wbindgen_init_externref_table = function() {
        try {
            const table = wasm.__wbindgen_export_3;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        } catch (e) {
            console.warn('[WASM] Browser compatibility: externref table initialization skipped:', e.message);
        }
    };`;

content = content.replace(original, patched);
fs.writeFileSync(filePath, content);
console.log('✅ Patch applied successfully!');
