import os
import subprocess
import time
import requests
import json
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class BitNet:
    def __init__(self):
        self.main_path = os.path.join("build", "bin", "llama-server")
        self.server_url = None
        self.slot_id: int = 1
        self.tokenizer = None
        self.streamer = None
        self.model = None
        self.quantized_model = None
        self.small_model = None
        self.pad_token = "<|pad|>"
        self.dtype = torch.float32
        self.process: Optional[subprocess.Popen] = None

    def __del__(self):
        self.stop_server()

    def start_server(
            self,
            bitnet_path: str,
            batch_size: int = 1,
            ctx_size: int = 1024,
            n_threads: int = 8,
            host: str = "127.0.0.1",
            port: int = 8080,
            extra_args: Optional[List[str]] = None,
            verbose: bool = False
    ):
        cmd = [
            self.main_path,
            "-m", bitnet_path,
            "--gpu-layers", "0",
            "-b", str(batch_size),
            "-c", str(ctx_size),
            "-t", str(n_threads),
            "-nkvo",  # disable KV offload
            "-fa",    # flash attention
            "--host", host,
            "--port", str(port),
        ]
        if verbose:
            cmd.append("-v")
        if extra_args:
            cmd += list(extra_args)

        print(f"🚀 Starting llama-server on {host}:{port}")
        self.server_url = f"http://{host}:{port}"
        self.quantized_model = bitnet_path
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
            tokenizer_id: str,
            verbose: bool = False
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            use_fast=True
        )
        self.tokenizer.pad_token = self.pad_token
        self.streamer = TextStreamer(self.tokenizer)

        if verbose:
            print(self.tokenizer)

    def init_model(
            self,
            model_path: str = None,
            device: str = "cpu",
            flash: bool = False,
            verbose: bool = False
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            dtype=self.dtype,
            attn_implementation="eager",
        )
        self.model.eval()

        if verbose:
            print(self.model)
            print(f"Number of parameters: {self.model.num_parameters() / 1e9:.2f}B")

    def encode_falcon_prompt(
            self,
            system_prompt: str,
            user_prompt: str
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True
        )
        return input_ids

    def generate_gguf(
            self,
            input_ids: int,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            top_k: int = 0,
            repeat_penalty: float = 1.0,
            slot_id: int = -1,
            seed: int = 42,
            verbose: bool = False,
    ):
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": input_ids,
            "n_predict": max_new_tokens,
            "cache_prompt": True,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": 0,
            "slot_id": slot_id,
            "seed": seed,
            "n_probs": 1,
            "return_tokens": True,
            "timings_per_token": True,
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/completions",
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("content")
            timings = response_data.get('timings')
            prob_list = response_data.get("completion_probabilities")

            result_list = []
            for prob in prob_list:
                result_list.append({
                    "tok_str": prob["probs"][0]["tok_str"],
                    "prob": float(prob["probs"][0]["prob"])
                })

            if verbose:
                input_tokens = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                print("\n\033[95m" + "─" * 50)
                print("🧠 Generation Info (BitNet GGUF)")
                print("─" * 50 + "\033[0m")
                print(f"\033[94m💬 User Input:\033[0m\n{response_data.get('prompt', input_tokens)}")
                print(f"\n\033[92m🟢 Generated Text:\033[0m\n{content}")
                print("\n\033[94m📊 Timings:\033[0m")
                print(f"├─ Prefill: {timings.get('prompt_per_token_ms', 0):.2f} ms/token, {timings.get('prompt_per_second', 0):.2f} tokens/s")
                print(f"└─ Decode: {timings.get('predicted_per_token_ms', 0):.2f} ms/token, {timings.get('predicted_per_second', 0):.2f} tokens/s")
                print(f"\033[94m📦 Tokens:\033[0m")
                print(f"├─ Prefilled: {response_data.get('tokens_evaluated', 0)}")
                print(f"└─ Decoded: {response_data.get('tokens_predicted', 0)}")
                print(f"\033[94m🛑 Stop Reason:\033[0m {response_data.get('stopping_word', 'N/A') or ('EOS' if response_data.get('stopped_eos') else 'Limit' if response_data.get('stopped_limit') else 'Unknown')}")
                print("\n\033[95m" + "─" * 50)
                print("💡 Token Probabilities")
                print("─" * 50 + "\033[0m")
                print(f"| {'Step':>4s} | {'Token':<15s} | {'Probability':>12s} |\033[0m")
                print(f"|{'-'*6}|{'-'*17}|{'-'*14}|\033[0m")
                for i, prob_info in enumerate(response_data['completion_probabilities']):
                    selected_tok_str = prob_info['content'].replace('\n', '\\n')
                    selected_prob = prob_info['probs'][0]['prob']
                    print(f"| {i+1:>4d} | {selected_tok_str:<15.15s} | {selected_prob:>12.2%} |\033[0m")

            return content, result_list

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")

    @torch.no_grad()
    def generate_hf(
            self,
            input_tokens: str,
            max_new_tokens: int = 100,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50,
            small_model=None,
            num_assistant_tokens: int = None,
            assistant_early_exit: int = None,
            stream: bool = False,
            verbose: bool = False,
    ):
        start_time = time.time()

        input_ids = self.tokenizer(
            input_tokens,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)
        num_input_tokens = input_ids["input_ids"].shape[-1]

        outputs = self.model.generate(
            **input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
            assistant_model=small_model,
            num_assistant_tokens=num_assistant_tokens,
            assistant_early_exit=assistant_early_exit,
            streamer=self.streamer if stream else None,
            return_dict_in_generate=True,
            output_scores=verbose
        )
        generated = outputs.sequences[0][num_input_tokens:]

        generated_text = self.tokenizer.decode(
            generated.tolist(),
            skip_special_tokens=False
        )

        end_time = time.time()
        total_time = end_time - start_time
        num_generated_tokens = len(generated)

        tokens_per_second = num_generated_tokens / total_time
        ms_per_token = (total_time * 1000) / num_generated_tokens if num_generated_tokens > 0 else 0

        if verbose:
            print("\n\033[95m" + "─" * 50)
            print("🧠 Generation Info (Hugging Face)")
            print("─" * 50 + "\033[0m")
            print(f"\033[94m💬 User Input:\033[0m\n{input_tokens}")
            print(f"\n\033[92m🟢 Generated Text:\033[0m\n{generated_text}")
            print("\n\033[94m📊 Timings:\033[0m")
            print(f"├─ Total Time: {total_time:.2f}s")
            print(f"├─ Latency: {ms_per_token:.2f} ms/token")
            print(f"└─ Throughput: {tokens_per_second:.2f} tokens/s")
            print(f"\033[94m📦 Tokens:\033[0m")
            print(f"├─ Prefilled: {num_input_tokens}")
            print(f"└─ Decoded: {num_generated_tokens}")
            
            print("\n\033[95m" + "─" * 50)
            print("💡 Token Probabilities")
            print("─" * 50 + "\033[0m")
            print(f"| {'Step':>4s} | {'Token':<15s} | {'Probability':>12s} |")
            print(f"|{'-'*6}|{'-'*17}|{'-'*14}|")
            main_scores = []
            for score_tuple in outputs.scores:
                main_scores.append(score_tuple[0] if isinstance(score_tuple, tuple) else score_tuple)
            probs = torch.stack([torch.softmax(s, dim=-1) for s in main_scores], dim=1).squeeze(0)
            for i, token_id in enumerate(generated):
                if i < probs.shape[0]:
                    token_prob = probs[i, token_id].item()
                    token_str = self.tokenizer.decode(token_id).replace('\n', '\\n')
                    print(f"| {i+1:>4d} | {token_str:<15.15s} | {token_prob:>12.2%} |")

    @torch.no_grad()
    def speculative_decode(
            self,
            input_ids: int,
            max_new_tokens: int,
            small_model = None,
            num_assistant_tokens: int = 10,
            seed: int = 42,
            verbose: bool = False,
    ):
        if verbose:
            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("✨ Speculative Decoding")
            print(f"├─ Target Model: {self.model.config.name_or_path}")
            if small_model is None:
                print(f"├─ Draft Model: {self.quantized_model}")
            else:
                print(f"├─ Draft Model: {small_model.config.name_or_path}")
            print(f"└─ Draft Length: {num_assistant_tokens}")
            print("\033[95m" + "─" * 50 + "\033[0m")

        start_time = time.time()

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
        generated_ids: list[int] = []

        step = 1
        total_draft_tokens = 0
        total_accepted_draft_tokens = 0

        # Generation loop
        while len(generated_ids) < max_new_tokens:
            if generated_ids:
                prefix = torch.cat([input_ids, torch.tensor([generated_ids], device=self.model.device)], dim=-1)
            else:
                prefix = input_ids

            # 1) Draft Generation
            if small_model is None: # GGUF
                draft, _ = self.generate_gguf(
                    input_ids=prefix[0].tolist(),
                    max_new_tokens=num_assistant_tokens,
                    temperature=0.0, top_p=1.0, min_p=0.0, top_k=0, seed=seed,
                    verbose=False
                )
                draft_ids = self.tokenizer(
                    draft, return_attention_mask=False, return_tensors="pt", add_special_tokens=False
                ).input_ids[0].to(self.model.device)
            else: # HF
                draft_ids = small_model.generate(
                    input_ids=prefix,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=num_assistant_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True
                ).sequences[0]
                draft_ids = draft_ids[prefix.shape[-1]:].to(self.model.device)
            total_draft_tokens += draft_ids.numel()

            # 2. Target Verification
            verification_ids = torch.cat([prefix, draft_ids.unsqueeze(0)], dim=-1)
            outputs = self.model(
                input_ids=verification_ids,
                use_cache=False,
                do_sample=False
            )
            logits = outputs.logits
            L = verification_ids.shape[-1] - draft_ids.shape[0]

            if verbose:
                print("\n" + "\033[95m" + "-" * 10 + f"Step {step}" + "-" * 10 + "\033[0m")
                print(f"┌{'─' * 5}┬{'─' * 17}┬{'─' * 17}┬{'─' * 17}┐")
                print(f"│ {'Idx':<3s} │ {'Draft Token':<15s} │ {'Status':<15s} │ {'Corrected':<15s} │")
                print(f"├{'─' * 5}┼{'─' * 17}┼{'─' * 17}┼{'─' * 17}┤")

            accepted_count = 0
            for i in range(draft_ids.shape[0]):
                target_logit = logits[:, L + i - 1, :] if (L + i - 1) >= 0 else logits[:, 0, :]

                draft_token_id = draft_ids[i].item()
                top1_id = torch.argmax(target_logit, dim=-1).item()

                accept = (top1_id == draft_token_id)
                if accept:
                    accepted_count += 1
                    if verbose:
                        tok_str = self.tokenizer.decode([draft_token_id]).replace('\n', '\\n')
                        print(f"│ {i + 1:<3d} │ {tok_str:<15.15s} │ \033[92m✅ Accepted\033[0m     │ {'-':<15s} │")
                else:
                    if accepted_count > 0:
                        generated_ids.extend(draft_ids[:accepted_count].tolist())
                        total_accepted_draft_tokens += accepted_count
                    corrected = top1_id
                    generated_ids.append(corrected)

                    if verbose:
                        tok_str = self.tokenizer.decode([draft_token_id]).replace('\n', '\\n')
                        corr_str = self.tokenizer.decode([corrected]).replace('\n', '\\n')
                        print(f"│ {i + 1:<3d} │ {tok_str:<15.15s} │ \033[91m ❌ Rejected\033[0m     │ {corr_str:<15.15s} │")
                        print(f"└{'─' * 5}┴{'─' * 17}┴{'─' * 17}┴{'─' * 17}┘")

                    break

            else: # Accepted all drafts
                total_accepted_draft_tokens += accepted_count
                generated_ids.extend(draft_ids.tolist())

                last_logit = logits[:, -1, :]
                next_token = torch.argmax(last_logit, dim=-1).item()
                generated_ids.append(next_token)

                if verbose:
                    next_token_str = self.tokenizer.decode([next_token]).replace('\n', '\\n')
                    print(f"└{'─' * 5}┴{'─' * 17}┴{'─' * 17}┴{'─' * 17}┘")
                    print(f"✅ \033[92mAdded 1 target token\033[0m: {next_token_str}")

            if self.tokenizer.eos_token_id in generated_ids:
                eos_index = generated_ids.index(self.tokenizer.eos_token_id)
                generated_ids = generated_ids[:eos_index]
                if verbose: print("🛑 EOS token generated. Stopping.")
                break

            if len(generated_ids) >= max_new_tokens:
                generated_ids = generated_ids[:max_new_tokens]
                break

            step += 1

        end_time = time.time()
        total_time = end_time - start_time

        acceptance_rate = (total_accepted_draft_tokens / total_draft_tokens) * 100 if total_draft_tokens > 0 else 0
        latency = (total_time / len(generated_ids)) * 1000 if len(generated_ids) > 0 else float('inf')
        throughput = len(generated_ids) / total_time if total_time > 0 else float('inf')

        input_tokens = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        final_text = self.tokenizer.batch_decode(torch.tensor(generated_ids, device=self.model.device).unsqueeze(0))[0]

        if verbose:
            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("🏁 Speculative Decoding Finished")
            print(f"\033[94m💬 Input:\033[0m\n{input_tokens}")
            print(f"\n\033[92m🟢 Generated:\033[0m\n{final_text}")
            print("\n\033[94m📊 Performance:\033[0m")
            print(f"├─ Total Time: {total_time:.2f}s")
            print(f"├─ Latency: {latency:.2f} ms/token")
            print(f"└─ Throughput: {throughput:.2f} tokens/s")
            print(f"\033[94m✨ Speculative Stats:\033[0m")
            print(f"├─ Total Generated Tokens: {len(generated_ids)}")
            print(f"├─ Acceptance Rate: {acceptance_rate:.2f}%")
            print(f"├─ Total Drafted Tokens: {total_draft_tokens}")
            print(f"└─ Total Accepted Draft Tokens: {total_accepted_draft_tokens}")

        return {
            "generated": final_text,
            "latency": latency,
            "acceptance_rate": acceptance_rate
        }