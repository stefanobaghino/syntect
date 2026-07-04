//! Trail-based speculation engine (`trail-engine` feature): checkpoint the
//! interpreter state before every branch decision, log the decision on a
//! trail, and handle `fail` by bumping the most recent live same-name
//! decision and deterministically re-executing the buffered window from
//! its checkpoint — instead of surgically correcting already-emitted ops
//! the way the legacy engine in `speculation.rs` does.
//!
//! The executor (`parse_next_token` and everything below it) is shared
//! with the legacy engine and operates on [`Core`]; the only trail-aware
//! points are branch-decision selection and `fail` handling.

use super::*;

/// Cap on checkpoint restarts within a single `parse_line` call, scaled
/// by the window size. Termination is guaranteed without it (every
/// restart lexicographically advances the finite decision trail), but
/// the worst case is exponential in the number of live decisions; the
/// budget converts a pathological syntax into a warning plus a committed
/// parse.
const RESTART_BUDGET_PER_WINDOW_LINE: usize = 64;

/// One branch decision: which alternative of the named `branch_point`
/// is currently on the stack, or `Refuse` when every alternative has
/// been exhausted and the branch pattern is suppressed at its trigger
/// position so the parent context's next rule fires (Sublime Text's
/// exhaustion semantics).
#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) enum Choice {
    Take(usize),
    Refuse,
}

/// Everything needed to resume execution from just before a branch
/// trigger match: the interpreter state plus the executor-local
/// bookkeeping of the line being executed. Restoring a checkpoint and
/// re-searching re-selects the branch trigger deterministically, so the
/// trigger's scopes, captures, and the chosen alternative's meta ops are
/// re-emitted through the ordinary `exec_pattern` path for *any*
/// alternative — nothing needs to be saved for manual re-emission.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Checkpoint {
    core: Core,
    /// Window-line index the decision sits on.
    line_idx: usize,
    /// Cursor position the branch trigger was matched from.
    pos: usize,
    /// Length of the line's ops vec before the trigger emitted anything.
    ops_len: usize,
    non_consuming_push_at: (usize, usize, usize),
    /// Branch suppressions active at this point of the line (from
    /// earlier `Refuse` decisions at not-yet-passed positions).
    skipped_branches: Vec<(usize, String)>,
    /// Live-branch set at decision time, exclusive of this decision.
    live: Vec<LiveBranch>,
}

/// A recorded branch decision plus the checkpoint to restart from when
/// a `fail` bumps it to its next alternative.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Decision {
    name: String,
    choice: Choice,
    num_alts: usize,
    /// Stack depth at creation (before the branch's own pops and the
    /// alternative's push).
    stack_depth: usize,
    /// `pop_count` of the branch operation; the alternative's frame
    /// lives at `stack_depth - pop_count + 1`.
    pop_count: usize,
    /// Absolute line number of creation, for 128-line expiry.
    created_line: usize,
    checkpoint: Checkpoint,
}

/// A decision whose alternative frame is still on the stack — the only
/// decisions a `fail` can target. Dropping an entry finalizes the
/// decision (it stays on the trail for deterministic re-execution).
#[derive(Debug, Clone, Eq, PartialEq)]
struct LiveBranch {
    decision_idx: usize,
    name: String,
    stack_depth: usize,
    pop_count: usize,
    created_line: usize,
}

impl LiveBranch {
    /// The alternative's frame is still on the stack iff the stack is
    /// deeper than the pre-branch depth minus the branch's own pops —
    /// the same predicate the legacy engine uses at its prune sites.
    fn alive(&self, stack_len: usize) -> bool {
        stack_len > self.stack_depth.saturating_sub(self.pop_count)
    }
}

/// The engine state: the buffered window, its single truth stream of
/// per-line ops, and the decision trail.
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub(super) struct TrailState {
    /// Absolute line number of `lines[0]`.
    base_line: usize,
    /// Buffered window: every line since the last commit point.
    lines: Vec<String>,
    /// Ops per window line — revised in place on re-execution.
    provisional: Vec<Vec<(usize, ScopeStackOp)>>,
    /// The decision log.
    trail: Vec<Decision>,
    /// Trail cursor during execution: a branch trigger encountered at
    /// the cursor consumes the recorded choice; past the end of the
    /// trail, new decisions are appended.
    cursor: usize,
    /// Decisions a `fail` can still target.
    live: Vec<LiveBranch>,
    /// Window-line index currently being executed.
    exec_line_idx: usize,
    /// Alternative selected by `decide_branch`, consumed by
    /// `exec_pattern`'s branch arm.
    pending_alt: Option<usize>,
    /// Set by `fail_branch`: decision index to restart from.
    pending_backtrack: Option<usize>,
    /// Restarts consumed by the current `parse_line` call.
    restarts: usize,
}
