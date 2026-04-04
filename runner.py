import io
from contextlib import redirect_stdout
from statistics import mean

from inference import MODEL_NAME, TASK_NAMES, EPISODES_PER_TASK, USE_LLM, run_episode


def main():
    model_label = MODEL_NAME if USE_LLM else "heuristic-baseline"
    print(f"Running baseline across tasks: {', '.join(TASK_NAMES)}")
    print(f"Episodes per task: {EPISODES_PER_TASK}")
    print(f"Agent mode: {'llm' if USE_LLM else 'heuristic'}")
    print("=" * 70)

    overall_scores = []
    for task_name in TASK_NAMES:
        task_scores = []
        print(f"\n===== TASK: {task_name.upper()} =====")
        for episode in range(1, EPISODES_PER_TASK + 1):
            print(f"\n[EPISODE] task={task_name} episode={episode}/{EPISODES_PER_TASK} model={model_label}")
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = run_episode(task_name, seed=1000 + episode + len(task_name))
            print(buf.getvalue().strip())
            print(
                f"[RESULT] grader_score={result['grader_score']:.2f} success={str(result['success']).lower()} steps={result['steps']}"
            )
            task_scores.append(result["grader_score"])
            overall_scores.append(result["grader_score"])
        print(f"\n[TASK SUMMARY] task={task_name} average_score={mean(task_scores):.2f} best_score={max(task_scores):.2f}")

    print("\n===== FINAL SUMMARY =====")
    print(f"Average score over {len(overall_scores)} episodes: {mean(overall_scores):.2f}")


if __name__ == "__main__":
    main()
