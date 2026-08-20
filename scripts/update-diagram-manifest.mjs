import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const sourceDir = new URL('diagrams/', root);
const outputDir = new URL('docs/assets/diagrams/', root);
const manifestURL = new URL('diagrams/manifest.json', root);
const sha256 = (content) => createHash('sha256').update(content).digest('hex');

// Usage:
//   node scripts/update-diagram-manifest.mjs <mermaidCliVersion> <svgoVersion>
//   node scripts/update-diagram-manifest.mjs --check   (verify only, never rewrite)
const args = process.argv.slice(2);
const checkMode = args[0] === '--check';
if (checkMode && args.length !== 1) {
  throw new Error('--check takes no additional arguments');
}
if (!checkMode && args.length !== 2) {
  throw new Error('expected exactly two renderer versions, e.g. `node scripts/update-diagram-manifest.mjs 11.16.0 4.0.2` (or `--check`)');
}

const names = (await readdir(sourceDir))
  .filter((name) => name.endsWith('.mmd'))
  .map((name) => name.slice(0, -4))
  .sort();

const files = await Promise.all(names.map(async (name) => {
  const [source, svg] = await Promise.all([
    readFile(new URL(`${name}.mmd`, sourceDir)),
    readFile(new URL(`${name}.svg`, outputDir)),
  ]);
  const viewBox = svg.toString('utf8').match(/viewBox="([^"]+)"/)?.[1];
  if (!viewBox) throw new Error(`${name}.svg has no viewBox`);
  return { name, sourceSha256: sha256(source), svgSha256: sha256(svg), viewBox };
}));

const manifest = {
  renderer: {
    mermaidCli: checkMode ? null : args[0],
    svgo: checkMode ? null : args[1],
  },
  files,
};

if (checkMode) {
  const current = JSON.parse(await readFile(manifestURL, 'utf8'));
  const problems = [];
  if (current.renderer?.mermaidCli === 'unknown' || current.renderer?.svgo === 'unknown') {
    problems.push('renderer versions missing (rerun with explicit versions)');
  }
  if (JSON.stringify(current.files) !== JSON.stringify(files)) {
    problems.push('MMD/SVG hashes or viewBox drift (rerun render-diagrams.sh, then regenerate)');
  }
  if (problems.length > 0) {
    console.error(`manifest check failed:\n  - ${problems.join('\n  - ')}`);
    process.exit(1);
  }
  console.log(`manifest check passed (${files.length} entries)`);
} else {
  await writeFile(manifestURL, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`manifest written (${files.length} entries)`);
}
