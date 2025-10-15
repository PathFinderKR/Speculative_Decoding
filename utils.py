import os
import subprocess
import time
import requests
import json
import random
import numpy as np
from typing import List, Dict, Optional
import torch
from transformers import TextStreamer

class BitNet:
    def __init__(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = 1024,
        n_threads: int = 12,
        n_gpu_layers: int = 0,
        batch_size: int = 1,
        slot_id: int = 1,
    ):
        self.main_path = os.path.join("build", "bin", "llama-server")
        self.model_path = model_path
        self.host = host
        self.port = int(port)
        self.ctx_size = int(ctx_size)
        self.n_threads = int(n_threads)
        self.n_gpu_layers = int(n_gpu_layers)
        self.batch_size = int(batch_size)
        self.slot_id = int(slot_id)
        self.server_url =  f"http://{self.host}:{self.port}"
        self.messages: List[Dict[str, str]] = []
        self.process: Optional[subprocess.Popen] = None

    def __del__(self):
        self.stop_server()

    def start_server(self, extra_args: Optional[List[str]] = None, verbose: bool = False):
        cmd = [
            self.main_path,
            "-m", self.model_path,
            "-c", str(self.ctx_size),
            "-t", str(self.n_threads),
            "-ngl", str(self.n_gpu_layers),
            "-b", str(self.batch_size),
            "--host", self.host,
            "--port", str(self.port),
            "-nkvo", # disable KV offload
        ]
        if verbose:
            cmd.append("-v")
        if extra_args:
            cmd += list(extra_args)

        print(f"🚀 Starting llama-server on {self.host}:{self.port}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for _ in range(10):
            try:
                response = requests.get(f"{self.server_url}/health")
                if response.status_code == 200 and response.json().get("status") == "ok":
                    print("✅ Server is ready.")
                    return

            except requests.ConnectionError:
                pass
            time.sleep(1)
        print("❌ Server failed to start in 10 seconds.")

    def stop_server(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            print("🛑 Server stopped.")

    def format_falcon_prompt(
            self,
            tokenizer,
            system_prompt: str,
            user_prompt: str,
            assistant_response: str = ""
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = input_ids + assistant_response
        return input_ids

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    @torch.no_grad()
    def generate(
            self,
            text: str,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            top_k: int = 0,
            seed: int = 42,
            verbose: bool = False,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": text,
            "n_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "seed": seed,
            "slot_id": self.slot_id,
            "n_probs": 1 if verbose else 0,
        }

        try:
            response = requests.post(
                f"{self.server_url}/completion",
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("content", "")
            if verbose:
                timings = response_data.get('timings', {})
                print("\n\033[95m" + "─" * 50)
                print("🧠 Generation Info")
                print("─" * 50 + "\033[0m")
                print(f"\033[94m💬 User Input:\033[0m\n{response_data.get('prompt', text)}")
                print(f"\n\033[92m🟢 Generated Text:\033[0m\n{content}")
                print("\n\033[94m📊 Timings:\033[0m")
                print(f"  - Prefill: {timings.get('prompt_per_token_ms', 0):.2f} ms/token, {timings.get('prompt_per_second', 0):.2f} tokens/s")
                print(f"  - Decode: {timings.get('predicted_per_token_ms', 0):.2f} ms/token, {timings.get('predicted_per_second', 0):.2f} tokens/s")
                print(f"\033[94m📦 Tokens:\033[0m")
                print(f"  - Prefilled: {response_data.get('tokens_evaluated', 0)}")
                print(f"  - Decoded: {response_data.get('tokens_predicted', 0)}")
                print(f"\033[94m🛑 Stop Reason:\033[0m {response_data.get('stopping_word', 'N/A') or ('EOS' if response_data.get('stopped_eos') else 'Limit' if response_data.get('stopped_limit') else 'Unknown')}")

                if 'completion_probabilities' in response_data and response_data['completion_probabilities']:
                    print("\n\033[95m" + "─" * 50)
                    print("💡 Token Probabilities")
                    print("─" * 50 + "\033[0m")
                    print(f"| {'Step':>4s} | {'Token':<15s} | {'Probability':>12s} |\033[0m")
                    print(f"|{'-'*6}|{'-'*17}|{'-'*14}|\033[0m")
                    for i, prob_info in enumerate(response_data['completion_probabilities']):
                        selected_tok_str = prob_info['content'].replace('\n', '\\n')
                        selected_prob = prob_info['probs'][0]['prob']
                        print(f"| {i+1:>4d} | {selected_tok_str:<15.15s} | {selected_prob:>12.2%} |\033[0m")
            return content

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return ""

    def verify(
            self,
            text: str,
            num_verify: int,
            threshold: float,
            verbose: bool = False
    ) -> Dict:
        """
        Verifies the last `num_verify` tokens of the given text using the /infill endpoint.

        Returns a dictionary with the index of the first rejected token (`cut_index`)
        and the list of probabilities for the verified tokens.
        """
        # 1. Tokenize the text to find the split point
        try:
            tokenize_response = requests.post(f"{self.server_url}/tokenize", json={"content": text})
            tokenize_response.raise_for_status()
            tokens = tokenize_response.json().get("tokens", [])
        except requests.exceptions.RequestException as e:
            print(f"Error during tokenization API request: {e}")
            return {"cut_index": 0, "probs": []}

        if not tokens:
            return {"cut_index": 0, "probs": []}

        if len(tokens) < num_verify:
            num_verify = len(tokens)

        if num_verify <= 0:
            return {"cut_index": 0, "probs": []}

        split_point = len(tokens) - num_verify

        # Reconstruct prefix and suffix from tokens to avoid character boundary issues
        try:
            prefix = requests.post(f"{self.server_url}/detokenize", json={"tokens": tokens[:split_point]}).json()[
                "content"]
            suffix = requests.post(f"{self.server_url}/detokenize", json={"tokens": tokens[split_point:]}).json()[
                "content"]
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"Error during detokenization: {e}")
            return {"cut_index": 0, "probs": []}

        # 2. Call /infill to get probabilities for the suffix
        headers = {"Content-Type": "application/json"}
        data = {
            "input_prefix": prefix,
            "input_suffix": suffix,
            "n_predict": 0,  # We only want to evaluate, not generate
            "slot_id": self.slot_id,
            "n_probs": 1,
        }

        try:
            response = requests.post(
                f"{self.server_url}/infill",
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()
            response_data = response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error during verification API request: {e}")
            return {"cut_index": 0, "probs": []}

        # 3. Process probabilities and find the cut_index
        probs: List[float] = []
        cut_index = num_verify

        completion_probs = response_data.get("completion_probabilities", [])

        for i, prob_info in enumerate(completion_probs):
            if i >= num_verify:
                break
            # The most probable token is the first in the list
            prob = prob_info['probs'][0]['prob']
            probs.append(prob)

            if prob < threshold and cut_index == num_verify:
                cut_index = i

        if verbose:
            print("\n\033[95m" + "─" * 50)
            print("🤖 Verification Report")
            print("─" * 50 + "\033[0m")
            print(f"\033[94mVerified Text:\033[0m\n{suffix}")
            print(f"\033[93mConfidence Threshold:\033[0m {threshold:.2f}\n")
            print("\033[96m🧩 Token Probabilities:\033[0m")
            for i, prob in enumerate(probs):
                tok_str = completion_probs[i]['content'].replace('\n', '\\n')
                status = (
                    "\033[92mACCEPTED\033[0m ✅" if prob >= threshold
                    else "\033[91mREJECTED\033[0m ❌"
                )
                print(f"  • Token[{i:02d}]: '{tok_str or '[SPACE]'}' → P={prob:.3f} → {status}")

            print(f"\n\033[93m✂️  Cut Index:\033[0m {cut_index}")
            print("\033[95m" + "─" * 50 + "\033[0m\n")

        return {"cut_index": cut_index, "probs": probs}





def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

@torch.no_grad()
def generate_response(
    tokenizer,
    model,
    assistant_model=None,
    system_prompt: str = "You are an helpful assistant.",
    user_prompt: str = "",
    max_new_tokens: int = 128,
    num_assistant_tokens: int = None,
    assistant_confidence_threshold: float = None,
    verbose: bool = True,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        use_cache=False,
        #cache_implementation="quantized",
        assistant_model=assistant_model,
        num_assistant_tokens=num_assistant_tokens,
        assistant_confidence_threshold=assistant_confidence_threshold,
        streamer=TextStreamer(tokenizer),
        return_dict_in_generate=verbose,
        output_scores=verbose,
    )

    if verbose:
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        generated_tokens = outputs.sequences[:, 1:]

        print("\n\033[95m─────────────────────────────")
        print("🧠 Token Generation Log")
        print("─────────────────────────────\033[0m")
        print(f"System Prompt: \033[94m{system_prompt}\033[0m")
        print(f"User Prompt:   \033[94m{user_prompt}\033[0m\n")

        print(f"\033[96m| {'#':>3s} | {'Token ID':>8s} | {'Token':<12s} | {'LogProb':>9s} | {'Prob':>9s} |\033[0m")
        print(f"\033[90m|{'-' * 5}|{'-' * 10}|{'-' * 14}|{'-' * 11}|{'-' * 11}|\033[0m")

        for i, (tok, score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
            prob = float(np.exp(score.numpy()))
            color = "\033[92m" if prob > 0.5 else ("\033[93m" if prob > 0.1 else "\033[91m")
            tok_text = tokenizer.decode(tok).replace("\n", "\\n").replace("\t", "\\t") or "[SPACE]"
            print(f"{color}| {i:3d} | {int(tok):8d} | {tok_text:<12.12s} | {score.numpy():9.3f} | {prob:9.2%} |\033[0m")

        print("\033[90m─────────────────────────────\033[0m\n")

        decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"\033[92m🟢 Final Generated Text:\033[0m\n{decoded_text}\n")
        print("\033[95m─────────────────────────────\033[0m\n")

        return None

    else:
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_text