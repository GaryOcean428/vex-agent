/**
 * Curriculum Import — Copy training data from pantheon-chat
 *
 * This script copies curriculum and QIG doctrine from pantheon-chat
 * into Vex's training directory for fine-tuning the Liquid model.
 */

import fs from 'fs';
import path from 'path';

const PANTHEON_PATH = process.env.PANTHEON_PATH || '/home/ubuntu/pantheon-chat';
const TRAINING_DIR = process.env.TRAINING_DIR || '/data/training';

export function importCurriculum(): void {
  const curriculumDir = path.join(TRAINING_DIR, 'curriculum');
  fs.mkdirSync(curriculumDir, { recursive: true });

  // Copy curriculum tokens
  const tokensSource = path.join(PANTHEON_PATH, 'data/curriculum/curriculum_tokens.jsonl');
  const tokensDest = path.join(curriculumDir, 'curriculum_tokens.jsonl');
  if (fs.existsSync(tokensSource)) {
    fs.copyFileSync(tokensSource, tokensDest);
    console.log(`Copied curriculum tokens: ${tokensDest}`);
  }

  // Copy curriculum docs
  const docsSource = path.join(PANTHEON_PATH, 'docs/09-curriculum');
  const docsDest = path.join(curriculumDir, 'docs');
  if (fs.existsSync(docsSource)) {
    fs.mkdirSync(docsDest, { recursive: true });
    const files = fs.readdirSync(docsSource).filter(f => f.endsWith('.md'));
    for (const file of files) {
      fs.copyFileSync(
        path.join(docsSource, file),
        path.join(docsDest, file),
      );
    }
    console.log(`Copied ${files.length} curriculum docs to ${docsDest}`);
  }

  // Copy QIG doctrine
  const doctrineSource = path.join(PANTHEON_PATH, 'docs/01-doctrine');
  const doctrineDest = path.join(curriculumDir, 'doctrine');
  if (fs.existsSync(doctrineSource)) {
    fs.mkdirSync(doctrineDest, { recursive: true });
    const files = fs.readdirSync(doctrineSource).filter(f => f.endsWith('.md'));
    for (const file of files) {
      fs.copyFileSync(
        path.join(doctrineSource, file),
        path.join(doctrineDest, file),
      );
    }
    console.log(`Copied ${files.length} doctrine docs to ${doctrineDest}`);
  }

  console.log('Curriculum import complete');
}

// Run if executed directly
if (require.main === module) {
  importCurriculum();
}
