import os
import subprocess
import time
import requests
import json
import random
import numpy as np
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class BitNet:
    def __init__(
        self,
        model_id: str,
        quantized_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = 1024,
        n_threads: int = 12,
        n_gpu_layers: int = 0,
        batch_size: int = 1,
        slot_id: int = 1,
    ):
        self.main_path = os.path.join("build", "bin", "llama-server")
        self.tokenizer = None
        self.streamer = None
        self.tokenizer_id = model_id
        self.pad_token = "<|pad|>"
        self.model = None
        self.model_id = model_id
        self.dtype = torch.float32
        self.model_path = quantized_path
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

    def init_tokenizer(
            self,
            verbose: bool = False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            use_fast=True,
        )
        self.tokenizer.pad_token = self.pad_token
        self.streamer = TextStreamer(self.tokenizer)

        if verbose:
            print(self.tokenizer)

    def init_model(
            self,
            verbose: bool = False
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            dtype=self.dtype
        )
        self.model.eval()

        if verbose:
            print(self.model)
            print(f"Number of parameters: {self.model.num_parameters() / 1e9:.2f}B")

    def format_falcon_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            assistant_response: str = ""
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
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
    def generate_hf(
            self,
            text: str,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_p: float = 1.0,
            speculative: bool = False,
            num_assistant_tokens: int = None,
            assistant_confidence_threshold: float = None,
            stream: bool = False,
            verbose: bool = False,
    ) -> str:
        quantized_model = None
        if speculative:
            quantized_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                gguf_file=self.model_path,
                device_map="cpu",
                dtype=self.dtype
            )
            quantized_model.eval()

        start_time = time.time()

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            use_cache=False,
            # cache_implementation="quantized",
            assistant_model=quantized_model if speculative else None,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
            streamer=self.streamer if not stream else None,
            return_dict_in_generate=True,
            output_scores=verbose,
        )
        generated = outputs.sequences[0][input_len:]
        generated_text = self.tokenizer.decode(generated.tolist(), skip_special_tokens=False)

        end_time = time.time()

        if verbose:
            total_time = end_time - start_time
            num_input_tokens = input_len
            num_generated_tokens = len(generated)

            decode_time = total_time
            decode_tokens_per_second = num_generated_tokens / decode_time if decode_time > 0 else float('inf')
            decode_ms_per_token = (decode_time * 1000) / num_generated_tokens if num_generated_tokens > 0 else 0

            print("\n\033[95m" + "─" * 50)
            print("🧠 Generation Info (Hugging Face)")
            print("─" * 50 + "\033[0m")
            print(f"\033[94m💬 User Input:\033[0m\n{text}")
            print(f"\n\033[92m🟢 Generated Text:\033[0m\n{generated_text}")
            print("\n\033[94m📊 Timings:\033[0m")
            print(f"  - Total Time: {total_time:.2f}s")
            print(f"  - Decode: {decode_ms_per_token:.2f} ms/token, {decode_tokens_per_second:.2f} tokens/s")
            print(f"\033[94m📦 Tokens:\033[0m")
            print(f"  - Prefilled: {num_input_tokens}")
            print(f"  - Decoded: {num_generated_tokens}")
            if speculative:
                print(f"\033[94m✨ Speculative Decoding:\033[0m")
                print(f"  - Assistant Tokens: {num_assistant_tokens}")
                print(f"  - Acceptance Rate: {outputs.acceptance_rate * 100:.2f}%")

            print("\n\033[95m" + "─" * 50)
            print("💡 Token Probabilities")
            print("─" * 50 + "\033[0m")
            print(f"| {'Step':>4s} | {'Token':<15s} | {'Probability':>12s} |")
            print(f"|{'-'*6}|{'-'*17}|{'-'*14}|")

            probs = torch.stack([torch.softmax(s, dim=-1) for s in outputs.scores], dim=1).squeeze()

            for i, token_id in enumerate(generated):
                token_prob = probs[i, token_id].item()
                token_str = self.tokenizer.decode(token_id).replace('\n', '\\n')
                print(f"| {i+1:>4d} | {token_str:<15.15s} | {token_prob:>12.2%} |")

        return generated_text

    @torch.no_grad()
    def generate_gguf(
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

    @torch.no_grad()
    def verify_hf(
            self,
            text: str,
            num_verify: int = 4,
            confidence_threshold: float = 0.4,
            verbose: bool = False
    ):
        # 1. Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        total_len = input_ids.shape[1]

        # 2. Forward pass
        outputs = self.model(input_ids)
        logits = outputs.logits

        # 3. Probabilities
        verify_logits = logits[0, -num_verify - 1:-1, :]
        probs = torch.softmax(verify_logits, dim=-1)
        verify_token_ids = input_ids[0, -num_verify:]

        if verbose:
            print("\n\033[95m" + "─" * 50)
            print("🔍 Verification Info")
            print("─" * 50 + "\033[0m")
            print(f"\033[94m📝 Full Text:\033[0m\n{text}")
            print(f"\033[94m🔢 Verifying Last {num_verify} Tokens:\033[0m {self.tokenizer.decode(verify_token_ids)}")
            print(f"\033[94m🎯 Confidence Threshold:\033[0m {confidence_threshold:.2%}")
            print("\n" + "─" * 50)
            header = f"| {'Step':>4s} | {'Token':<15s} | {'Probability':>12s} | {'Status':<10s} |"
            print(header)
            print(f"|{'-'*6}|{'-'*17}|{'-'*14}|{'-'*12}|")

        accepted = True
        results = []
        for i in range(num_verify):
            token_id = verify_token_ids[i]
            token_prob = probs[i, token_id].item()

            if accepted and token_prob >= confidence_threshold:
                status = "Accepted"
                status_color = "\033[92m"
            else:
                accepted = False
                status = "Rejected"
                status_color = "\033[91m"

            token_str = self.tokenizer.decode(token_id).replace('\n', '\\n')
            results.append({"token": token_str, "prob": token_prob, "status": status})

            if verbose:
                print(f"| {total_len - num_verify + i + 1:>4d} | {token_str:<15.15s} | {token_prob:>12.2%} | {status_color}{status:<10s}\033[0m |")

        if verbose:
            print("\033[95m" + "─" * 50 + "\033[0m")

        return results

    def speculative_decoding(
            self,
            text: str,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_p: float = 1.0,
            top_k: int = 0,
            num_assistant_tokens: int = 4,
            confidence_threshold: float = 0.4,
            verbose: bool = False,
    ):
        full_generated_text = ""
        current_text = text
        n_generated = 0
        step = 1
        total_time_start = time.time()

        if verbose:
            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("✨ Starting Speculative Decoding")
            print(f"├─ Target Model: {self.model_id}")
            print(f"├─ Draft Model: {self.model_path}")
            print(f"└─ Draft Length: {num_assistant_tokens}, Confidence: {confidence_threshold:.1f}")
            print("\033[95m" + "─" * 50 + "\033[0m")

        while n_generated < max_new_tokens:
            # 1. Draft Generation (Fast Model)
            remaining_tokens = max_new_tokens - n_generated
            draft_len = min(num_assistant_tokens, remaining_tokens)

            if verbose:
                print(f"\n\033[96m--- Step {step} ---\033[0m")
                print(f"📝 \033[94mPrompting draft model with:\033[0m\n...{current_text[-50:]}")

            draft_text = self.generate_gguf(
                text=current_text,
                max_new_tokens=draft_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                verbose=False,
            )

            if not draft_text:
                if verbose: print("⚠️ Draft model produced no output. Stopping.")
                break

            draft_token_ids = self.tokenizer.encode(draft_text, add_special_tokens=False)
            num_draft_tokens = len(draft_token_ids)

            if num_draft_tokens == 0:
                if verbose: print("⚠️ Draft model produced no new tokens. Stopping.")
                break

            if verbose:
                print(f"🚀 \033[93mDraft generated:\033[0m\n{draft_text.strip()} ({num_draft_tokens} tokens)")

            # 2. Verification (Accurate Model)
            verification_text = current_text + draft_text
            verification_results = self.verify_hf(
                text=verification_text,
                num_verify=num_draft_tokens,
                confidence_threshold=confidence_threshold,
                verbose=verbose,
            )

            # 3. Accept/Reject Logic
            n_accepted = 0
            for result in verification_results:
                if result["status"] == "Accepted":
                    n_accepted += 1
                else:
                    break  # First rejection, stop here

            accepted_ids = draft_token_ids[:n_accepted]
            accepted_text = self.tokenizer.decode(accepted_ids)

            if verbose:
                print(f"✅ \033[92mAccepted {n_accepted} tokens:\033[0m '{accepted_text.strip()}'")

            # Append accepted tokens to the context for the next step
            current_text += accepted_text
            full_generated_text += accepted_text
            n_generated += n_accepted

            if n_generated >= max_new_tokens:
                break

            # If not all draft tokens were accepted, perform a correction step
            if n_accepted < num_draft_tokens:
                if verbose:
                    print("🤔 \033[91mPerforming correction step...\033[0m")

                # Generate a single token with the accurate model
                correction_token = self.generate_hf(
                    text=current_text,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_p=top_p,
                    verbose=False,
                )

                if not correction_token or self.tokenizer.eos_token in correction_token:
                    if verbose: print("⏹️ Correction step produced EOS or was empty. Stopping.")
                    break

                if verbose:
                    print(f"👍 \033[92mCorrected token:\033[0m '{correction_token.strip()}'")

                current_text += correction_token
                full_generated_text += correction_token
                n_generated += 1

            if self.tokenizer.eos_token in accepted_text:
                if verbose: print("⏹️ EOS token found in accepted text. Stopping.")
                break

            step += 1

        total_time_end = time.time()
        total_time = total_time_end - total_time_start

        if verbose:
            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("🏁 Speculative Decoding Finished")
            print(f"├─ Total Generated: {n_generated} tokens")
            if total_time > 0:
                print(f"└─ Total Time: {total_time:.2f}s ({n_generated / total_time:.2f} tokens/s)")
            else:
                print(f"└─ Total Time: {total_time:.2f}s")
            print("\033[95m" + "─" * 50 + "\033[0m")

        return full_generated_text


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