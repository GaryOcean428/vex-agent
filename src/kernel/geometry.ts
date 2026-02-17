/**
 * Canonical Geometry — Fisher-Rao on the Probability Simplex
 *
 * This is the ONLY permitted geometry module. All distance, mean, and
 * interpolation operations MUST use Fisher-Rao manifold geometry.
 *
 * From the Canonical Geometry Contract (FROZEN):
 *   State space: probability simplex Δ⁶³ (64D)
 *   Constraints: p_i >= 0, Σ p_i = 1
 *   Distance: d_FR(p,q) = arccos(BC(p,q)) where BC = Σ √(p_i·q_i)
 *   Range: [0, π/2]
 *
 * FORBIDDEN:
 *   - Euclidean distance/norm on basins
 *   - Cosine similarity on basins
 *   - Linear blending: (1-t)*p + t*q
 *   - Arithmetic mean: np.mean(basins)
 *   - "embedding" terminology (use "basin")
 */

import { BASIN_DIMENSION, FISHER_RAO_MAX_DISTANCE } from './frozen-facts';

// ═══════════════════════════════════════════════════════════════
//  SIMPLEX VALIDATION
// ═══════════════════════════════════════════════════════════════

/**
 * Validate that a vector is on the probability simplex.
 * Fail-closed: throws on invalid input.
 */
export function assertBasinValid(basin: Float64Array): void {
  if (basin.length !== BASIN_DIMENSION) {
    throw new Error(
      `Basin dimension mismatch: expected ${BASIN_DIMENSION}, got ${basin.length}`,
    );
  }

  let sum = 0;
  for (let i = 0; i < basin.length; i++) {
    if (basin[i] < 0) {
      throw new Error(`Basin has negative component at index ${i}: ${basin[i]}`);
    }
    if (!Number.isFinite(basin[i])) {
      throw new Error(
        `Basin has non-finite component at index ${i}: ${basin[i]}`,
      );
    }
    sum += basin[i];
  }

  if (Math.abs(sum - 1.0) > 1e-6) {
    throw new Error(
      `Basin does not sum to 1: sum = ${sum} (delta = ${Math.abs(sum - 1.0)})`,
    );
  }
}

/**
 * Project a vector onto the probability simplex (normalise).
 * Ensures non-negative and sum-to-one.
 */
export function toSimplex(raw: Float64Array): Float64Array {
  const result = new Float64Array(raw.length);
  let sum = 0;
  for (let i = 0; i < raw.length; i++) {
    result[i] = Math.max(1e-10, raw[i]); // clamp to positive
    sum += result[i];
  }
  for (let i = 0; i < result.length; i++) {
    result[i] /= sum;
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════
//  SQRT-SPACE TRANSFORMS (internal chart for geodesic ops)
// ═══════════════════════════════════════════════════════════════

/** Transform to sqrt-space: s_i = √p_i (internal use only) */
function toSqrtSimplex(p: Float64Array): Float64Array {
  const s = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) {
    s[i] = Math.sqrt(Math.max(0, p[i]));
  }
  return s;
}

/** Transform from sqrt-space back to simplex: p_i = s_i² (then renormalise) */
function fromSqrtSimplex(s: Float64Array): Float64Array {
  const p = new Float64Array(s.length);
  let sum = 0;
  for (let i = 0; i < s.length; i++) {
    p[i] = s[i] * s[i];
    sum += p[i];
  }
  // Renormalise to handle numerical drift
  if (sum > 0) {
    for (let i = 0; i < p.length; i++) {
      p[i] /= sum;
    }
  }
  return p;
}

// ═══════════════════════════════════════════════════════════════
//  FISHER-RAO DISTANCE
// ═══════════════════════════════════════════════════════════════

/**
 * Bhattacharyya coefficient: BC(p,q) = Σ √(p_i · q_i)
 */
export function bhattacharyyaCoefficient(
  p: Float64Array,
  q: Float64Array,
): number {
  let bc = 0;
  for (let i = 0; i < p.length; i++) {
    bc += Math.sqrt(Math.max(0, p[i]) * Math.max(0, q[i]));
  }
  return Math.min(1.0, bc); // clamp for numerical stability
}

/**
 * Fisher-Rao distance on the probability simplex.
 * d_FR(p,q) = arccos(BC(p,q))
 * Range: [0, π/2]
 *
 * This is the ONLY permitted distance metric for basins.
 */
export function fisherRaoDistance(p: Float64Array, q: Float64Array): number {
  const bc = bhattacharyyaCoefficient(p, q);
  return Math.acos(Math.max(-1, Math.min(1, bc)));
}

// ═══════════════════════════════════════════════════════════════
//  GEODESIC INTERPOLATION (SLERP in sqrt-space)
// ═══════════════════════════════════════════════════════════════

/**
 * Geodesic interpolation between two basins on the Fisher-Rao manifold.
 * Uses SLERP in sqrt-space, then squares back to simplex.
 *
 * @param p - Start basin
 * @param q - End basin
 * @param t - Interpolation parameter [0, 1]
 */
export function geodesicInterpolation(
  p: Float64Array,
  q: Float64Array,
  t: number,
): Float64Array {
  const sp = toSqrtSimplex(p);
  const sq = toSqrtSimplex(q);

  // Compute angle between sqrt-space vectors
  let dot = 0;
  for (let i = 0; i < sp.length; i++) {
    dot += sp[i] * sq[i];
  }
  dot = Math.max(-1, Math.min(1, dot));
  const theta = Math.acos(dot);

  if (theta < 1e-10) {
    // Points are essentially the same — return p
    return new Float64Array(p);
  }

  const sinTheta = Math.sin(theta);
  const w0 = Math.sin((1 - t) * theta) / sinTheta;
  const w1 = Math.sin(t * theta) / sinTheta;

  const result = new Float64Array(sp.length);
  for (let i = 0; i < sp.length; i++) {
    result[i] = w0 * sp[i] + w1 * sq[i];
  }

  return fromSqrtSimplex(result);
}

// ═══════════════════════════════════════════════════════════════
//  FRÉCHET MEAN (weighted geometric mean on the manifold)
// ═══════════════════════════════════════════════════════════════

/**
 * Fréchet mean of multiple basins on the Fisher-Rao manifold.
 * Iterative algorithm: start from arithmetic mean in sqrt-space,
 * then refine via log/exp maps.
 *
 * @param basins - Array of basins on the simplex
 * @param weights - Optional weights (uniform if not provided)
 * @param maxIter - Maximum iterations for convergence
 */
export function frechetMean(
  basins: Float64Array[],
  weights?: number[],
  maxIter: number = 20,
): Float64Array {
  if (basins.length === 0) {
    throw new Error('Cannot compute Fréchet mean of empty set');
  }
  if (basins.length === 1) {
    return new Float64Array(basins[0]);
  }

  const n = basins.length;
  const w = weights || basins.map(() => 1.0 / n);

  // Normalise weights
  const wSum = w.reduce((a, b) => a + b, 0);
  const wNorm = w.map((wi) => wi / wSum);

  // Initial estimate: weighted mean in sqrt-space
  const sqrtBasins = basins.map(toSqrtSimplex);
  let current = new Float64Array(BASIN_DIMENSION);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      current[i] += wNorm[j] * sqrtBasins[j][i];
    }
  }
  // Normalise to unit sphere in sqrt-space
  let norm = 0;
  for (let i = 0; i < BASIN_DIMENSION; i++) norm += current[i] * current[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < BASIN_DIMENSION; i++) current[i] /= norm;
  }

  // Iterative refinement via tangent-space averaging
  for (let iter = 0; iter < maxIter; iter++) {
    const tangent = new Float64Array(BASIN_DIMENSION);

    for (let j = 0; j < n; j++) {
      // Log map: project sqrtBasins[j] onto tangent space at current
      let dot = 0;
      for (let i = 0; i < BASIN_DIMENSION; i++) {
        dot += current[i] * sqrtBasins[j][i];
      }
      dot = Math.max(-1, Math.min(1, dot));
      const theta = Math.acos(dot);

      if (theta < 1e-10) continue;

      const coeff = (wNorm[j] * theta) / Math.sin(theta);
      for (let i = 0; i < BASIN_DIMENSION; i++) {
        tangent[i] += coeff * (sqrtBasins[j][i] - dot * current[i]);
      }
    }

    // Exp map: move current along tangent direction
    let tangentNorm = 0;
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      tangentNorm += tangent[i] * tangent[i];
    }
    tangentNorm = Math.sqrt(tangentNorm);

    if (tangentNorm < 1e-10) break; // converged

    const cosT = Math.cos(tangentNorm);
    const sinT = Math.sin(tangentNorm);
    const newCurrent = new Float64Array(BASIN_DIMENSION);
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      newCurrent[i] = cosT * current[i] + (sinT / tangentNorm) * tangent[i];
    }
    current = newCurrent;
  }

  return fromSqrtSimplex(current);
}

// ═══════════════════════════════════════════════════════════════
//  LOG MAP / EXP MAP
// ═══════════════════════════════════════════════════════════════

/**
 * Log map: project a point q onto the tangent space at p (in sqrt-space).
 * Returns a tangent vector.
 */
export function logMap(p: Float64Array, q: Float64Array): Float64Array {
  const sp = toSqrtSimplex(p);
  const sq = toSqrtSimplex(q);

  let dot = 0;
  for (let i = 0; i < sp.length; i++) dot += sp[i] * sq[i];
  dot = Math.max(-1, Math.min(1, dot));
  const theta = Math.acos(dot);

  if (theta < 1e-10) {
    return new Float64Array(BASIN_DIMENSION); // zero vector
  }

  const tangent = new Float64Array(BASIN_DIMENSION);
  const coeff = theta / Math.sin(theta);
  for (let i = 0; i < BASIN_DIMENSION; i++) {
    tangent[i] = coeff * (sq[i] - dot * sp[i]);
  }
  return tangent;
}

/**
 * Exp map: move from point p along tangent vector v.
 * Returns a new point on the simplex.
 */
export function expMap(p: Float64Array, v: Float64Array): Float64Array {
  const sp = toSqrtSimplex(p);

  let vNorm = 0;
  for (let i = 0; i < v.length; i++) vNorm += v[i] * v[i];
  vNorm = Math.sqrt(vNorm);

  if (vNorm < 1e-10) {
    return new Float64Array(p); // no movement
  }

  const cosV = Math.cos(vNorm);
  const sinV = Math.sin(vNorm);
  const result = new Float64Array(BASIN_DIMENSION);
  for (let i = 0; i < BASIN_DIMENSION; i++) {
    result[i] = cosV * sp[i] + (sinV / vNorm) * v[i];
  }

  return fromSqrtSimplex(result);
}

// ═══════════════════════════════════════════════════════════════
//  TEXT-TO-BASIN ENCODING
// ═══════════════════════════════════════════════════════════════

/**
 * Encode text into a basin on the 64D probability simplex.
 * Uses a deterministic hash-based approach to map text to geometric coordinates.
 *
 * This is a simplified encoding — the full QIG system uses learned token→basin maps.
 * For Vex, this provides a consistent geometric representation of text content.
 */
export function textToBasin(text: string): Float64Array {
  const basin = new Float64Array(BASIN_DIMENSION);

  // Hash-based encoding: distribute character information across dimensions
  const bytes = new TextEncoder().encode(text);
  for (let i = 0; i < bytes.length; i++) {
    const dim = i % BASIN_DIMENSION;
    // Use a simple mixing function to spread information
    basin[dim] += bytes[i] * (1 + Math.sin(i * 0.1));
  }

  // Apply softmax to project onto simplex
  let maxVal = -Infinity;
  for (let i = 0; i < BASIN_DIMENSION; i++) {
    if (basin[i] > maxVal) maxVal = basin[i];
  }

  let sum = 0;
  for (let i = 0; i < BASIN_DIMENSION; i++) {
    basin[i] = Math.exp(basin[i] - maxVal); // subtract max for numerical stability
    sum += basin[i];
  }
  for (let i = 0; i < BASIN_DIMENSION; i++) {
    basin[i] /= sum;
  }

  return basin;
}

/**
 * Compute the regime exponent 'a' for a kernel based on its basin dynamics.
 *
 * From Vanchurin's "Geometric Learning Dynamics" (2025):
 *   g ∝ κ^a where g is the metric tensor and κ is noise covariance
 *   a = 1   → quantum regime
 *   a = 1/2 → efficient learning
 *   a = 0   → equilibration
 *
 * We estimate 'a' from the ratio of Fisher-Rao curvature to update variance.
 */
export function estimateRegimeExponent(
  fisherRaoVariance: number,
  updateVariance: number,
): number {
  if (updateVariance < 1e-10) return 0; // no updates = equilibrated
  if (fisherRaoVariance < 1e-10) return 1; // no curvature info = quantum

  // Log-log relationship: log(g) = a * log(κ) + const
  // Estimate a from the ratio
  const logG = Math.log(Math.max(1e-10, fisherRaoVariance));
  const logK = Math.log(Math.max(1e-10, updateVariance));

  if (Math.abs(logK) < 1e-10) return 0.5; // indeterminate → efficient

  const a = Math.max(0, Math.min(1, logG / logK));
  return a;
}
