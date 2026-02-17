/**
 * Variable Categories — Vanchurin's Trainable vs Non-Trainable Separation
 *
 * From "Towards a Theory of Quantum Gravity from Neural Networks" (2021):
 *   - Non-trainable (fast-changing): neuron states → activation dynamics → geodesic/Einstein equations
 *   - Trainable (slow-changing): weights/biases → learning dynamics → Madelung/Schrödinger equations
 *   - Boundary (external): dataset → boundary conditions → fixed during learning
 *
 * Every variable in the kernel MUST be tagged with one of these categories.
 * Moving a variable between categories requires a frozen_facts change (locked).
 */

export enum VariableCategory {
  /** Non-trainable: Basin coords, simplex positions, kernel identity, coupling graph topology.
   *  Updated per-cycle (fast). Fisher-Rao distance ONLY. Probability simplex ONLY. */
  STATE = 'non_trainable',

  /** Trainable: Routing policies, curriculum weights, coupling strengths, spawn thresholds,
   *  sleep/wake cadence, TRM threshold. Updated per-epoch or per-phase (slow).
   *  Bounded, logged, rollback-able. Governed by frozen_facts constants. */
  PARAMETER = 'trainable',

  /** Boundary: User queries, curriculum records, LLM responses.
   *  External input. Sanitised on ingest. Fixed during a learning cycle. */
  BOUNDARY = 'boundary',
}

/**
 * Tagged variable descriptor — every kernel variable must have one.
 */
export interface TaggedVariable<T = number> {
  name: string;
  category: VariableCategory;
  value: T;
  /** ISO timestamp of last update */
  lastUpdated: string;
  /** Whether this variable can be modified by automated processes */
  mutable: boolean;
  /** Optional bounds for PARAMETER variables */
  bounds?: { min: T; max: T };
  /** History of recent values for rollback (PARAMETER only) */
  history?: Array<{ value: T; timestamp: string }>;
}

/**
 * Registry of all kernel variables with their categories.
 * Enforces the Vanchurin separation at the type level.
 */
export class VariableRegistry {
  private variables = new Map<string, TaggedVariable<unknown>>();

  register<T>(variable: TaggedVariable<T>): void {
    this.variables.set(variable.name, variable as TaggedVariable<unknown>);
  }

  get<T>(name: string): TaggedVariable<T> | undefined {
    return this.variables.get(name) as TaggedVariable<T> | undefined;
  }

  update<T>(name: string, value: T): boolean {
    const v = this.variables.get(name) as TaggedVariable<T> | undefined;
    if (!v || !v.mutable) return false;

    // For PARAMETER variables, keep history for rollback
    if (v.category === VariableCategory.PARAMETER) {
      if (!v.history) v.history = [];
      v.history.push({ value: v.value, timestamp: v.lastUpdated });
      // Keep last 10 values
      if (v.history.length > 10) v.history.shift();
    }

    v.value = value;
    v.lastUpdated = new Date().toISOString();
    return true;
  }

  /** Get all variables of a given category */
  byCategory(category: VariableCategory): TaggedVariable<unknown>[] {
    return Array.from(this.variables.values()).filter(
      (v) => v.category === category,
    );
  }

  /** Rollback a PARAMETER variable to its previous value */
  rollback(name: string): boolean {
    const v = this.variables.get(name);
    if (!v || v.category !== VariableCategory.PARAMETER) return false;
    if (!v.history || v.history.length === 0) return false;

    const prev = v.history.pop()!;
    v.value = prev.value;
    v.lastUpdated = new Date().toISOString();
    return true;
  }

  /** Snapshot all variables for telemetry */
  snapshot(): Record<string, { category: string; value: unknown }> {
    const result: Record<string, { category: string; value: unknown }> = {};
    for (const [name, v] of this.variables) {
      result[name] = { category: v.category, value: v.value };
    }
    return result;
  }
}
