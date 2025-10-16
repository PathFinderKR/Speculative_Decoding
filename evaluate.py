import torch
from datasets import load_dataset
from tqdm import tqdm
from speculative_decoding import BitNet

class LlmEvaluator:
    def __init__(self, bitnet_instance: BitNet):
        """
        Initializes the evaluator with a BitNet instance.

        Args:
            bitnet_instance (BitNet): An initialized BitNet object with a model and tokenizer.
        """
        self.bitnet = bitnet_instance
        if not self.bitnet.model or not self.bitnet.tokenizer:
            raise ValueError("BitNet instance must have an initialized model and tokenizer.")

    def evaluate_hellaswag(self, split: str = "validation", num_samples: int = 100):
        """
        Evaluates the base model on the HellaSwag dataset.

        This method calculates the perplexity for each of the four possible endings
        and chooses the one with the lowest perplexity.

        Args:
            split (str): The dataset split to use (e.g., "validation").
            num_samples (int): The number of samples to evaluate from the dataset.

        Returns:
            dict: A dictionary containing the accuracy score.
        """
        print(f" HellaSwag 평가를 시작합니다 (모델: {self.bitnet.model_id}, 샘플 수: {num_samples}) ...")
        dataset = load_dataset("hellaswag", split=split)
        dataset = dataset.select(range(num_samples))

        correct_predictions = 0
        total = 0

        for i, item in enumerate(tqdm(dataset, desc="HellaSwag Evaluation")):
            context = item['ctx']
            endings = item['endings']
            label = int(item['label'])

            perplexities = []
            for ending in endings:
                full_text = context + " " + ending
                try:
                    ppl = self.bitnet.calculate_perplexity_hf(full_text)
                    perplexities.append(ppl)
                except Exception as e:
                    print(f"Error calculating perplexity for item {i}: {e}")
                    perplexities.append(float('inf'))

            # 가장 낮은 perplexity를 가진 보기 선택
            prediction = torch.argmin(torch.tensor(perplexities)).item()

            if prediction == label:
                correct_predictions += 1
            total += 1

        accuracy = (correct_predictions / total) * 100 if total > 0 else 0
        print(f" HellaSwag 평가 완료. 정확도: {accuracy:.2f}%")
        return {"accuracy": accuracy}

    def evaluate_wikitext_perplexity(self, split: str = "test", num_samples: int = 10):
        """
        Evaluates the base model's perplexity on the WikiText-2 dataset.

        Args:
            split (str): The dataset split to use (e.g., "test").
            num_samples (int): The number of samples (articles) to evaluate.

        Returns:
            dict: A dictionary containing the mean perplexity score.
        """
        print(f" WikiText-2 Perplexity 평가를 시작합니다 (모델: {self.bitnet.model_id}, 샘플 수: {num_samples}) ...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        total_ppl = 0
        count = 0

        for i in tqdm(range(num_samples), desc="WikiText-2 PPL Evaluation"):
            text = dataset[i]['text']
            if text.strip():  # 비어있지 않은 텍스트만 평가
                try:
                    ppl = self.bitnet.calculate_perplexity_hf(text)
                    total_ppl += ppl
                    count += 1
                except Exception as e:
                    print(f"Error calculating perplexity for sample {i}: {e}")

        mean_perplexity = total_ppl / count if count > 0 else float('inf')
        print(f" WikiText-2 Perplexity 평가 완료. 평균 Perplexity: {mean_perplexity:.4f}")
        return {"perplexity": mean_perplexity}