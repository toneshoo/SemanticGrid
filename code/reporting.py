from tqdm import tqdm


def print_gnn_training_summary(train_cases, seconds, device, accuracy, voltage_mae):
    print(f"GNN trained on {train_cases} cases in {seconds:.0f}s ({device}): test accuracy "
          f"{accuracy:.1%}, voltage MAE {voltage_mae:.4f} p.u.")


def print_active_learning_pool_summary(pool_size, hidden_count, kind, warm_start, weights):
    warm_note = f", warm start from {weights}" if warm_start else ""
    print(f"pool: {pool_size:,} candidate {kind}, {hidden_count} hidden dangerous "
          f"({hidden_count / pool_size:.2%}){warm_note}", flush=True)


def print_active_learning_run_summary(strategy, seed, warm, found, dangerous, kind, recall, physics_calls):
    warm_note = " warm" if warm else ""
    tqdm.write(f"{strategy:<18} seed {seed}{warm_note}: {found}/{dangerous} hidden {kind} "
               f"({recall:.1%}) with {physics_calls} ANDES solves")


def print_cegis_round(iteration, state_count, agreement, counterexample_count):
    print(f"round {iteration}: {state_count} states, {agreement:.2%} agreement with ANDES, "
          f"{counterexample_count} counterexamples")


def print_verification_summary(state_count, n1_agreement, holdout_agreement, exhaustive_agreement,
                               detection, confirmed_remediations, remediation_count):
    print(f"\nFSM: {state_count} states | agreement with ANDES: N-1 {n1_agreement:.1%}, "
          f"held-out N-2 {holdout_agreement:.1%}, all N-1 + N-2 {exhaustive_agreement:.1%}")
    print(f"N-2: {detection['n2_dangerous_pairs']} dangerous pairs, recall {detection['n2_recall']:.1%}, "
          f"precision {detection['n2_precision']:.1%} | hidden pairs {detection['hidden_pairs']}, "
          f"flagged {detection['hidden_pairs_flagged']} | corrective trips confirmed "
          f"{confirmed_remediations}/{remediation_count}")


def print_benchmark_summary(grid_name, sample_count, nr_ms, gnn_ms, gpu_batched_ms, fsm_ms, n3_hours):
    batched = f", GNN GPU batched {gpu_batched_ms:.4f} ms" if gpu_batched_ms is not None else ""
    print(f"{grid_name}, median of {sample_count}: Newton-Raphson {nr_ms:.2f} ms, "
          f"GNN CPU {gnn_ms:.2f} ms{batched}, FSM {fsm_ms:.5f} ms | N-3 sweep with NR: {n3_hours:.1f} h")


def print_n3_candidates(grid_name, candidate_count, islanding_count, remaining_count, core_count):
    print(f"{grid_name}: {candidate_count:,} candidate triples ({islanding_count:,} islanding removed), "
          f"{remaining_count:,} left to solve on {core_count} cores")


def print_n3_progress(done, total, minutes_left):
    print(f"  {done:,}/{total:,} solved, {minutes_left:.0f} min left", flush=True)


def print_n3_summary(state_counts, output_path):
    print(f"done: {state_counts} (everything except NORMAL is a hidden triple) -> {output_path}")
