#!/usr/bin/env python3
"""
Generate reasonably random arithmetic expressions with big integers.

Key characteristics:
- First generates a flat "a op b op c ..." expression string (no tree).
- Then injects parentheses with probability, mainly where they matter (near '*').
- Rejection-samples until abs(result) is under a max digit limit.

Only operators: +, -, *
"""

from __future__ import annotations

import argparse
import ast
import operator as op
import secrets
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


_rng = secrets.SystemRandom()


# --------------------------
# Safe evaluator (no eval)
# --------------------------

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

def safe_eval_expr(expr: str) -> int:
    """
    Safely evaluate an arithmetic expression containing only integers,
    parentheses, +, -, *.
    """
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, int):
                return n.value
            raise ValueError("Only integer constants are allowed.")
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](_eval(n.left), _eval(n.right))

        raise ValueError(f"Disallowed syntax: {ast.dump(n, include_attributes=False)}")

    return _eval(node)


# --------------------------
# Generator
# --------------------------

def digits_in_abs(x: int) -> int:
    x = abs(x)
    # str(0) -> "0" => 1 digit
    return len(str(x))


def rand_int_with_digits(min_digits: int, max_digits: int) -> int:
    d = _rng.randint(min_digits, max_digits)
    first = _rng.randint(1, 9)
    if d == 1:
        return first
    rest = "".join(str(_rng.randint(0, 9)) for _ in range(d - 1))
    return int(f"{first}{rest}")


def weighted_choice(items: List[Tuple[str, int]]) -> str:
    total = sum(w for _, w in items)
    r = _rng.randrange(total)
    acc = 0
    for sym, w in items:
        acc += w
        if r < acc:
            return sym
    return items[-1][0]  # fallback


@dataclass
class ParenConfig:
    # Probability to add parentheses around a "+/-" pair when adjacent to '*'
    local_pair_prob: float = 0.55
    # Max number of parentheses pairs to inject
    max_pairs: int = 2
    # Small chance to add a wider span (3-5 numbers) for variety
    wide_span_prob: float = 0.22
    wide_span_min_numbers: int = 3
    wide_span_max_numbers: int = 5


def build_flat_tokens(n_ops: int, num_digits_min: int, num_digits_max: int,
                      op_weights: List[Tuple[str, int]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (nums, ops, tokens_without_parens)
    tokens_without_parens looks like [num0, op0, num1, op1, num2, ...]
    """
    nums: List[str] = [
        str(rand_int_with_digits(num_digits_min, num_digits_max))
        for _ in range(n_ops + 1)
    ]
    ops: List[str] = [weighted_choice(op_weights) for _ in range(n_ops)]

    tokens: List[str] = []
    for i in range(n_ops):
        tokens.append(nums[i])
        tokens.append(ops[i])
    tokens.append(nums[-1])
    return nums, ops, tokens


def tokens_to_string(tokens: List[str]) -> str:
    return " ".join(tokens)


def inject_parentheses(tokens: List[str], ops: List[str], pcfg: ParenConfig) -> List[str]:
    """
    Add parentheses by editing the token list in-place style (returns new list).
    Parentheses are mostly added around '+/-' pairs adjacent to '*', which is where
    parentheses are "most necessary" to change precedence.

    We avoid heavy nesting by limiting to pcfg.max_pairs and by preventing overlap.
    """
    # Helper mapping: number index k lives at token index 2*k (ignoring parentheses insertions).
    # But once we insert parentheses, indices shift; so we will compute insertion points
    # based on current token positions by scanning.

    out = tokens[:]

    # Track number-token positions dynamically by scanning.
    def current_number_token_indices(tok_list: List[str]) -> List[int]:
        idxs = []
        for i, t in enumerate(tok_list):
            if t.isdigit():  # all numbers are positive digit strings in tokens
                idxs.append(i)
        return idxs

    # Build candidate "+/-" operator positions by op index (between num i and num i+1).
    # Condition: adjacent to '*' on left or right (in the ops list).
    candidates: List[int] = []
    for i, sym in enumerate(ops):
        if sym in ("+", "-"):
            left_is_mul = (i - 1 >= 0 and ops[i - 1] == "*")
            right_is_mul = (i + 1 < len(ops) and ops[i + 1] == "*")
            if left_is_mul or right_is_mul:
                candidates.append(i)

    _rng.shuffle(candidates)

    used_num_ranges: List[Tuple[int, int]] = []  # (start_num_i, end_num_j) inclusive

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] < b[0] or b[1] < a[0])

    pairs_added = 0

    # 1) Local pair parentheses: wrap (num_i op_i num_{i+1})
    for op_i in candidates:
        if pairs_added >= pcfg.max_pairs:
            break
        if _rng.random() > pcfg.local_pair_prob:
            continue

        span = (op_i, op_i + 1)  # number indices
        if any(overlaps(span, s) for s in used_num_ranges):
            continue

        num_token_idxs = current_number_token_indices(out)
        if op_i + 1 >= len(num_token_idxs):
            continue

        left_tok = num_token_idxs[op_i]
        right_tok = num_token_idxs[op_i + 1]

        # Insert '(' before left number and ')' after right number.
        # Insert ')' first so it doesn't affect the earlier index.
        out.insert(right_tok + 1, ")")
        out.insert(left_tok, "(")

        used_num_ranges.append(span)
        pairs_added += 1

    # 2) Optional wide-span parentheses (3-5 numbers), only if we still have budget.
    # We try to pick a span that includes at least one '*' and one '+/-' to be meaningful.
    if pairs_added < pcfg.max_pairs and _rng.random() < pcfg.wide_span_prob:
        max_numbers = min(pcfg.wide_span_max_numbers, len(ops) + 1)
        min_numbers = min(pcfg.wide_span_min_numbers, max_numbers)
        if min_numbers >= 2:
            span_len = _rng.randint(min_numbers, max_numbers)  # number count in span
            start_num = _rng.randint(0, (len(ops) + 1) - span_len)
            end_num = start_num + span_len - 1  # inclusive number index

            # Check overlap with existing local pairs
            span_nums = (start_num, end_num)
            if not any(overlaps(span_nums, s) for s in used_num_ranges):
                # Determine operator slice inside span
                inside_ops = ops[start_num:end_num]
                has_mul = "*" in inside_ops
                has_addsub = any(x in ("+", "-") for x in inside_ops)

                if has_mul and has_addsub:
                    num_token_idxs = current_number_token_indices(out)
                    left_tok = num_token_idxs[start_num]
                    right_tok = num_token_idxs[end_num]

                    out.insert(right_tok + 1, ")")
                    out.insert(left_tok, "(")
                    pairs_added += 1

    return out


@dataclass
class GenConfig:
    ops_min: int = 3
    ops_max: int = 10
    num_digits_min: int = 3
    num_digits_max: int = 7
    max_result_digits: int = 16
    max_attempts: int = 2000

    # Operator weights: increase + and - relative to * to reduce blow-ups
    w_add: int = 4
    w_sub: int = 3
    w_mul: int = 6

    paren: ParenConfig = field(default_factory=ParenConfig)


def generate_expression(cfg: GenConfig) -> Tuple[str, int]:
    op_weights = [("+", cfg.w_add), ("-", cfg.w_sub), ("*", cfg.w_mul)]

    for _ in range(cfg.max_attempts):
        n_ops = _rng.randint(cfg.ops_min, cfg.ops_max)

        nums, ops, flat_tokens = build_flat_tokens(
            n_ops=n_ops,
            num_digits_min=cfg.num_digits_min,
            num_digits_max=cfg.num_digits_max,
            op_weights=op_weights,
        )

        tokens = inject_parentheses(flat_tokens, ops, cfg.paren)
        expr = tokens_to_string(tokens)

        try:
            value = safe_eval_expr(expr)
        except Exception:
            continue  # should be rare; just resample

        if digits_in_abs(value) <= cfg.max_result_digits:
            return expr, value

    raise RuntimeError(
        f"Failed to generate an expression within {cfg.max_attempts} attempts "
        f"under {cfg.max_result_digits} digits. Consider lowering '*' weight or max ops."
    )


# --------------------------
# CLI
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate random big-integer arithmetic expressions.")
    p.add_argument("--cnt", type=int, default=10)

    defaults = GenConfig()

    p.add_argument("--ops-min", type=int, default=defaults.ops_min)
    p.add_argument("--ops-max", type=int, default=defaults.ops_max)
    p.add_argument("--digits-min", type=int, default=defaults.num_digits_min)
    p.add_argument("--digits-max", type=int, default=defaults.num_digits_max)
    p.add_argument("--max-result-digits", type=int, default=defaults.max_result_digits)
    p.add_argument("--attempts", type=int, default=defaults.max_attempts)

    p.add_argument("--w-add", type=int, default=defaults.w_add)
    p.add_argument("--w-sub", type=int, default=defaults.w_sub)
    p.add_argument("--w-mul", type=int, default=defaults.w_mul)

    p.add_argument("--paren-prob", type=float, default=defaults.paren.local_pair_prob, help="Probability for local pair parentheses near '*'")
    p.add_argument("--max-parens", type=int, default=defaults.paren.max_pairs, help="Maximum parentheses pairs to insert")
    p.add_argument("--wide-paren-prob", type=float, default=defaults.paren.wide_span_prob, help="Probability for adding one wider span")
    p.add_argument("--print-value", action="store_true")

    return p.parse_args()


def main() -> None:
    a = parse_args()

    cfg = GenConfig(
        ops_min=a.ops_min,
        ops_max=a.ops_max,
        num_digits_min=a.digits_min,
        num_digits_max=a.digits_max,
        max_result_digits=a.max_result_digits,
        max_attempts=a.attempts,
        w_add=a.w_add,
        w_sub=a.w_sub,
        w_mul=a.w_mul,
        paren=ParenConfig(
            local_pair_prob=a.paren_prob,
            max_pairs=a.max_parens,
            wide_span_prob=a.wide_paren_prob,
        ),
    )

    for _ in range(a.cnt):
        expr, value = generate_expression(cfg)
        expr = expr.replace("( ", "(").replace(" )", ")")
        if a.print_value:
            print(f"{expr}, {value}")
        else:
            print(expr)


if __name__ == "__main__":
    main()
