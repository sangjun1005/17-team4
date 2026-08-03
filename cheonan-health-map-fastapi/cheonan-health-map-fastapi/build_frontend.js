const fs = require('fs');
const path = require('path');
const ts = require('/opt/nvm/versions/node/v22.16.0/lib/node_modules/typescript');
const projectRoot = __dirname;
const sourceRoot = path.join(projectRoot, 'frontend_source');
const staticRoot = path.join(projectRoot, 'app', 'static');
const sourceFiles = [
  'app/page.tsx',
  'app/cheonan-boundary.ts',
  'app/vulnerability-summary.ts',
  'app/region-catalog.ts',
  'app/region-dashboard-data.ts',
  'app/region-supplemental-data.ts',
  'app/cheonan-age-population.ts',
];
const modules = {};
for (const relativePath of sourceFiles) {
  const source = fs.readFileSync(path.join(sourceRoot, relativePath), 'utf8');
  const result = ts.transpileModule(source, {
    fileName: relativePath,
    compilerOptions: {
      target: ts.ScriptTarget.ES2019,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
      allowSyntheticDefaultImports: true,
      resolveJsonModule: true,
      isolatedModules: true,
    },
    reportDiagnostics: true,
  });
  const errors = (result.diagnostics || []).filter((d) => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) {
    for (const d of errors) console.error(`${relativePath}: ${ts.flattenDiagnosticMessageText(d.messageText, '\n')}`);
    process.exit(1);
  }
  modules[relativePath.replace(/\.(tsx?|jsx?)$/, '.js').replace(/\\/g, '/')] = result.outputText;
}
const subdivisions = JSON.parse(fs.readFileSync(path.join(sourceRoot, 'app', 'cheonan-subdivisions.json'), 'utf8'));
const csv = fs.readFileSync(path.join(sourceRoot, 'app', 'data', 'cheonan-age-population-2026-06.csv'), 'utf8');
const wrap = (id, code) => `${JSON.stringify(id)}: function(module, exports, require) {\n${code}\n}`;
const entries = Object.entries(modules).map(([id, code]) => wrap(id, code));
entries.push(wrap('app/cheonan-subdivisions.json', `module.exports = ${JSON.stringify(subdivisions)};`));
entries.push(wrap('app/data/cheonan-age-population-2026-06.csv?raw', `module.exports = ${JSON.stringify(csv)};`));
const bundle = `(() => {
'use strict';
const modules = {\n${entries.join(',\n')}\n};
const cache = {};
function normalize(parts) { const out=[]; for (const p of parts) { if(!p||p==='.') continue; if(p==='..') out.pop(); else out.push(p); } return out.join('/'); }
function resolveRequest(parentId, request) {
  if (!request.startsWith('.')) return request;
  const parts=parentId.split('/'); parts.pop();
  const raw=normalize(parts.concat(request.split('/')));
  for (const id of [raw, raw+'.js', raw+'.json']) if (Object.prototype.hasOwnProperty.call(modules,id)) return id;
  throw new Error('모듈을 찾을 수 없습니다: '+request+' (from '+parentId+')');
}
const jsxRuntime = {
  Fragment: React.Fragment,
  jsx(type, props, key) { const p=props?{...props}:{}; if(key!==undefined)p.key=key; return React.createElement(type,p); },
  jsxs(type, props, key) { const p=props?{...props}:{}; if(key!==undefined)p.key=key; return React.createElement(type,p); }
};
function externalModule(id) {
  if(id==='react') return React;
  if(id==='react/jsx-runtime') return jsxRuntime;
  if(id==='leaflet') return window.L;
  throw new Error('지원하지 않는 외부 모듈입니다: '+id);
}
function requireModule(id) {
  if(!Object.prototype.hasOwnProperty.call(modules,id)) return externalModule(id);
  if(cache[id]) return cache[id].exports;
  const module={exports:{}}; cache[id]=module;
  modules[id](module,module.exports,(request)=>requireModule(resolveRequest(id,request)));
  return module.exports;
}
const pageModule=requireModule('app/page.js');
const Page=pageModule.default||pageModule;
ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(Page));
})();\n`;
fs.mkdirSync(staticRoot,{recursive:true});
fs.writeFileSync(path.join(staticRoot,'app.bundle.js'),bundle,'utf8');
const originalCss=fs.readFileSync(path.join(sourceRoot,'app','globals.css'),'utf8');
const preflightPath='/opt/nvm/versions/node/v22.16.0/lib/node_modules/tailwindcss/preflight.css';
const preflight=fs.existsSync(preflightPath)?fs.readFileSync(preflightPath,'utf8'):'';
const css=preflight+'\n'+originalCss.replace('@import "tailwindcss";','').replace('@import "leaflet/dist/leaflet.css";','');
fs.writeFileSync(path.join(staticRoot,'app.css'),css,'utf8');
console.log(`Built ${Object.keys(modules).length+2} modules`);
