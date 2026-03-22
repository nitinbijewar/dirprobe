# === Physics constants (FROZEN in v1.x — Governance Rule 7a) ===
# Mathematical basis: defines axis well-definedness criterion.
# Do NOT change in v1.x — alters which sites contribute to principal-axis metrics.
EPS_DEGENERACY = 0.05        # Manuscript default (§2.2, SI §S7).
                              # Reference implementation used 0.01; both produce
                              # identical masks on KNN data (min gap = 0.0607).
                              # Parity tests call 0.01 explicitly when matching
                              # the reference; all other code uses this default.

# === Robustness/classification defaults (LOCKED defaults, overridable — Rule 7b) ===
# Engineering judgment calls. Every function that uses these accepts them as kwargs.
# Users studying different material systems may override at call time.
RHO_PASS = 0.9               # Spearman PASS threshold (Test 1)
RHO_FLAG = 0.7               # Spearman FLAG/FAIL boundary (Test 1)
POOLING_PASS = 0.05          # Pooling sensitivity PASS (Test 2)
POOLING_FLAG = 0.10          # Pooling sensitivity FLAG boundary (Test 2)
TEMPORAL_PASS = 0.3          # D_dir CV PASS threshold (Test 3)
TEMPORAL_FLAG = 0.6          # D_dir CV FLAG boundary (Test 3)
PERSISTENCE_LOCKED = 0.8     # A and B threshold for LOCKED (manuscript §2.3)
PERSISTENCE_FAST = 0.4       # A threshold for FAST (manuscript §2.3)
MIN_SEGMENT_FRAMES = 10      # Minimum frames per segment for temporal test
DEFAULT_WINDOW_SIZE = 50     # Default window size for windowed alignment
DEFAULT_N_BLOCKS = 2         # Default number of blocks for block-to-full alignment
DEFAULT_MIN_BLOCK_SIZE = 50  # Minimum block size; shorter -> B = np.nan
