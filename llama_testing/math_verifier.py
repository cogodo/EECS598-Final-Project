# math_verifier.py

from verl.utils.reward_score.gsm8k import extract_solution, compute_score

class MathVerifier:
    """
    A simple wrapper around verl's GSM8K verifier utilities.
    Provides strict or flexible extraction and binary scoring.
    """

    def __init__(self, method="flexible", correct_reward=1.0, format_reward=0.0):
        """
        Args:
            method: "strict" or "flexible"
            correct_reward: reward value when answer matches ground truth
            format_reward: reward value when answer is formatted correctly but wrong
        """
        self.method = method
        self.correct_reward = correct_reward
        self.format_reward = format_reward

    def verify(self, prompt, response, ground_truth):
        """
        Returns:
            {
                "correct": bool,
                "reward": float,
                "parsed_answer": str or None
            }
        """
        parsed = extract_solution(response, method=self.method)

        score = compute_score(
            solution_str=response,
            ground_truth=str(ground_truth),
            method=self.method,
            score=self.correct_reward,
            format_score=self.format_reward,
        )

        return {
            "correct": score == self.correct_reward,
            "reward": float(score),
            "parsed_answer": parsed,
        }


if __name__ == "__main__":

    dummy_prompt = "Annie has 8 apples. A friend took 2 apples. How many apples does Annie have left?"

    # GSM8K ground truth format:
    dummy_ground_truth = "6"

    # Fake model output (free-form)
    dummy_model_output = "Annie has 6 apples left."

    mv = MathVerifier(method="flexible")

    result = mv.verify(
        prompt=dummy_prompt,
        response=dummy_model_output,
        ground_truth=dummy_ground_truth,
    )

    print("Prompt:", dummy_prompt)
    print("Model Output:", dummy_model_output)
    print("Ground Truth:", dummy_ground_truth)
    print("\nVerifier Result:")
    print(result)
