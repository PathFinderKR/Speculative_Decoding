import os
import subprocess
import time
import requests
import json
from typing import List, Dict, Optional
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
        self.messages: List[Dict[str, str]] = []
        self.process: Optional[subprocess.Popen] = None

    def __del__(self):
        self.stop_server()

    def start_server(
            self,
            bitnet_path: str,
            ctx_size: int = 1024,
            n_threads: int = 12,
            batch_size: int = 1,
            host: str = "127.0.0.1",
            port: int = 8080,
            extra_args: Optional[List[str]] = None,
            verbose: bool = False
    ):
        cmd = [
            self.main_path,
            "-m", bitnet_path,
            "-c", str(ctx_size),
            "-t", str(n_threads),
            "-ngl", "0",
            "-b", str(batch_size),
            "--host", host,
            "--port", str(port),
            "-nkvo", # disable KV offload
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
            flash: bool = False,
            verbose: bool = False
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            dtype=self.dtype,
            output_attentions=True,
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

    @torch.no_grad()
    def generate_hf(
            self,
            text: str,
            max_new_tokens: int = 128,
            stream: bool = False,
            verbose: bool = False,
    ) -> str:
        start_time = time.time()

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            streamer=self.streamer if stream else None,
            return_dict_in_generate=True,
            output_scores=verbose
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
            print(f"├─ Total Time: {total_time:.2f}s")
            print(f"└─ Decode: {decode_ms_per_token:.2f} ms/token, {decode_tokens_per_second:.2f} tokens/s")
            print(f"\033[94m📦 Tokens:\033[0m")
            print(f"├─ Prefilled: {num_input_tokens}")
            print(f"└─ Decoded: {num_generated_tokens}")
            
            print("\n\033[95m" + "─" * 50)
            print("💡 Token Probabilities")
            print("─" * 50 + "\033[0m")
            print(f"| {'Step':>4s} | {'Token':<15s} | {'Probability':>12s} |")
            print(f"|{'-'*6}|{'-'*17}|{'-'*14}|")
            main_scores = []
            if hasattr(outputs, "scores") and outputs.scores is not None:
                for score_tuple in outputs.scores:
                    main_scores.append(score_tuple[0] if isinstance(score_tuple, tuple) else score_tuple)
            if main_scores:
                probs = torch.stack([torch.softmax(s, dim=-1) for s in main_scores], dim=1).squeeze(0)
                for i, token_id in enumerate(generated):
                    if i < probs.shape[0]:
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
    ):
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
                print("🧠 Generation Info (BitNet GGUF)")
                print("─" * 50 + "\033[0m")
                print(f"\033[94m💬 User Input:\033[0m\n{response_data.get('prompt', text)}")
                print(f"\n\033[92m🟢 Generated Text:\033[0m\n{content}")
                print("\n\033[94m📊 Timings:\033[0m")
                print(f"├─ Prefill: {timings.get('prompt_per_token_ms', 0):.2f} ms/token, {timings.get('prompt_per_second', 0):.2f} tokens/s")
                print(f"└─ Decode: {timings.get('predicted_per_token_ms', 0):.2f} ms/token, {timings.get('predicted_per_second', 0):.2f} tokens/s")
                print(f"\033[94m📦 Tokens:\033[0m")
                print(f"├─ Prefilled: {response_data.get('tokens_evaluated', 0)}")
                print(f"└─ Decoded: {response_data.get('tokens_predicted', 0)}")
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

    @torch.no_grad()
    def speculative_decoding(
            self,
            text: str,
            max_new_tokens: int,
            num_assistant_tokens: int,
            confidence_threshold: float,
            seed: int = 42,
            verbose: bool = False,
    ):
        if verbose:
            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("✨ Speculative Decoding")
            print(f"├─ Target Model: {self.model.config.name_or_path}")
            print(f"├─ Draft Model: {self.quantized_model}")
            print(f"└─ Draft Length: {num_assistant_tokens}, Confidence: {confidence_threshold:.1f}")
            print("\033[95m" + "─" * 50 + "\033[0m")

        start_time = time.time()
        step = 1
        total_draft_tokens = 0
        total_accepted_draft_tokens = 0

        # Initial prompt
        prompt_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.model.device)
        generated_token_ids = []
        past_key_values = None

        # Generation loop
        while len(generated_token_ids) < max_new_tokens:
            current_input_ids = torch.cat(
                [
                    prompt_tokens,
                    torch.tensor([generated_token_ids], dtype=torch.long, device=self.model.device)
                ],
                dim=-1,
            )
            current_text = self.tokenizer.decode(current_input_ids[0])

            # 1. Draft Generation
            draft_text = self.generate_gguf(
                text=current_text,
                max_new_tokens=num_assistant_tokens,
                temperature=0.0,
                top_p=1.0,
                min_p=0.0,
                top_k=0,
                seed=seed,
                verbose=False,
            )
            draft_ids = self.tokenizer(
                draft_text,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0].to(self.model.device)

            if len(draft_ids) == 0:
                if verbose: print("⚠️ Draft model produced no new tokens. Stopping.")
                break
            total_draft_tokens += len(draft_ids)

            # 2. Target Verification
            verification_ids = torch.cat([
                current_input_ids,
                draft_ids.unsqueeze(0)
            ], dim=-1)
            outputs = self.model(
                input_ids=verification_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            if verbose:
                print("\n" + "\033[95m" + "-" * 10 + f"Step {step}" + "-" * 10 + "\033[0m")
                print(f"\033[94mDraft Input:\033[0m\n{current_text}")
                print(f"\033[94mDraft Output\033[0m\n{self.tokenizer.decode(draft_ids.tolist(), skip_special_tokens=False)}")
                print(f"┌{'─'*5}┬{'─'*17}┬{'─'*14}┬{'─'*20}┬{'─'*17}┐")
                print(f"│ {'Idx':<3s} │ {'Draft Token':<15s} │ {'Target Prob':>12s} │ {'Status':<18s} │ {'Corrected':<15s} │")
                print(f"├{'─'*5}┼{'─'*17}┼{'─'*14}┼{'─'*20}┼{'─'*17}┤")

            accepted_count = 0
            for i in range(len(draft_ids)):
                # 1. Probabilities
                target_logit = logits[:, current_input_ids.shape[-1] + i - 1, :]
                probs = torch.softmax(target_logit, dim=-1)

                draft_token_id = draft_ids[i]
                draft_token_prob = probs[0, draft_token_id].item()
                draft_token_str = self.tokenizer.decode(draft_token_id).replace('\n', '\\n')

                # 2. Accept/Reject
                if draft_token_prob >= confidence_threshold:
                    accepted_count += 1
                    if verbose:
                        status = "\033[92m✅ Accepted\033[0m"
                        corrected_str = "-"
                        print(f"│ {i + 1:<3d} │ {draft_token_str:<15.15s} │ {draft_token_prob:>12.2%} │ {status:<26s} │ {corrected_str:<15s} │")

                else:
                    corrected_token = torch.argmax(target_logit, dim=-1).item()
                    if verbose:
                        corrected_str = self.tokenizer.decode(corrected_token).replace('\n', '\\n')
                        status = "\033[91m❌ Rejected\033[0m"
                        print(f"│ {i + 1:<3d} │ {draft_token_str:<15.15s} │ {draft_token_prob:>12.2%} │ {status:<26s} │ {corrected_str:<15.15s} │")
                        print(f"└{'─' * 5}┴{'─' * 17}┴{'─' * 14}┴{'─' * 20}┴{'─' * 17}┘")
                    if accepted_count > 0:
                        generated_token_ids.extend(draft_ids[:accepted_count].tolist())
                    generated_token_ids.append(corrected_token)
                    total_accepted_draft_tokens += accepted_count
                    break
            else: # All draft tokens accepted
                total_accepted_draft_tokens += accepted_count
                generated_token_ids.extend(draft_ids.tolist())
                last_logit = logits[:, -1, :]
                next_token = torch.argmax(last_logit, dim=-1).item()
                generated_token_ids.append(next_token)
                if verbose:
                    print(f"└{'─' * 5}┴{'─' * 17}┴{'─' * 14}┴{'─' * 20}┴{'─' * 17}┘")
                    accepted_token_strs = [self.tokenizer.decode(t).replace('\n', '\\n') for t in draft_ids.tolist()]
                    next_token_str = self.tokenizer.decode(next_token).replace('\n', '\\n')
                    print(f"✅ \033[92mAccepted all {accepted_count} tokens: \033[0m{accepted_token_strs}")
                    print(f"✅ \033[92mGenerated Target Token: \033[0m{next_token_str}")

            if self.tokenizer.eos_token_id in generated_token_ids:
                eos_index = generated_token_ids.index(self.tokenizer.eos_token_id)
                generated_token_ids = generated_token_ids[:eos_index]
                if verbose: print("🛑 EOS token generated. Stopping.")
                break

            step += 1

        end_time = time.time()
        final_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)

        if verbose:
            total_time = end_time - start_time
            num_generated = len(generated_token_ids)
            acceptance_rate = (total_accepted_draft_tokens / total_draft_tokens) * 100 if total_draft_tokens > 0 else 0
            latency = (total_time / num_generated) * 1000 if num_generated > 0 else float('inf')
            throughput = num_generated / total_time if total_time > 0 else float('inf')

            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("🏁 Speculative Decoding Finished")
            print(f"\033[94m💬 User Input:\033[0m\n{text}")
            print(f"\n\033[92m🟢 Generated Text:\033[0m\n{final_text}")
            print("\n\033[94m📊 Performance:\033[0m")
            print(f"├─ Total Time: {total_time:.2f}s")
            print(f"├─ Latency: {latency:.2f} ms/token")
            print(f"└─ Throughput: {throughput:.2f} tokens/s")
            print(f"\033[94m✨ Speculative Stats:\033[0m")
            print(f"├─ Acceptance Rate: {acceptance_rate:.2f}%")
            print(f"├─ Total Drafted Tokens: {total_draft_tokens}")
            print(f"└─ Total Accepted Draft Tokens: {total_accepted_draft_tokens}")

        return final_text

    @torch.no_grad()
    def speculative_decoding_hf(
            self,
            small_model,
            text: str,
            max_new_tokens: int,
            num_assistant_tokens: int,
            confidence_threshold: float,
            verbose: bool = False,
    ):
        if verbose:
            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("✨ Speculative Decoding")
            print(f"├─ Target Model: {self.model.config.name_or_path}")
            print(f"├─ Draft Model: {small_model.config.name_or_path}")
            print(f"└─ Draft Length: {num_assistant_tokens}, Confidence: {confidence_threshold:.1f}")
            print("\033[95m" + "─" * 50 + "\033[0m")

        start_time = time.time()
        step = 1
        total_draft_tokens = 0
        total_accepted_draft_tokens = 0

        # Initial prompt
        prompt_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.model.device)
        generated_token_ids = []
        past_key_values = None

        # Generation loop
        while len(generated_token_ids) < max_new_tokens:
            current_input_ids = torch.cat(
                [
                    prompt_tokens,
                    torch.tensor([generated_token_ids], dtype=torch.long, device=self.model.device)
                ],
                dim=-1,
            )

            # 1. Draft Generation
            draft_outputs = small_model.generate(
                input_ids=current_input_ids.to(self.model.device),
                max_new_tokens=num_assistant_tokens,
                do_sample=False,
                use_cache=True
            )
            draft_ids = draft_outputs[0, current_input_ids.shape[-1]:].to(self.model.device)

            if len(draft_ids) == 0:
                if verbose: print("⚠️ Draft model produced no new tokens. Stopping.")
                break
            total_draft_tokens += len(draft_ids)

            # 2. Target Verification
            verification_ids = torch.cat([current_input_ids, draft_ids.unsqueeze(0)], dim=-1)
            outputs = self.model(
                input_ids=verification_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            if verbose:
                current_text = self.tokenizer.decode(current_input_ids[0])
                print("\n" + "\033[95m" + "-" * 10 + f"Step {step}" + "-" * 10 + "\033[0m")
                print(f"\033[94mDraft Input:\033[0m\n{current_text}")
                print(f"\033[94mDraft Output\033[0m\n{self.tokenizer.decode(draft_ids.tolist(), skip_special_tokens=False)}")
                print(f"┌{'─' * 5}┬{'─' * 17}┬{'─' * 14}┬{'─' * 20}┬{'─' * 17}┐")
                print(f"│ {'Idx':<3s} │ {'Draft Token':<15s} │ {'Target Prob':>12s} │ {'Status':<18s} │ {'Corrected':<15s} │")
                print(f"├{'─' * 5}┼{'─' * 17}┼{'─' * 14}┼{'─' * 20}┼{'─' * 17}┤")

            accepted_count = 0
            for i in range(len(draft_ids)):
                # 1. Probabilities
                target_logit = logits[:, current_input_ids.shape[-1] + i - 1, :]
                probs = torch.softmax(target_logit, dim=-1)

                draft_token_id = draft_ids[i]
                draft_token_prob = probs[0, draft_token_id].item()
                draft_token_str = self.tokenizer.decode(draft_token_id).replace('\n', '\\n')

                # 2. Accept/Reject
                if draft_token_prob >= confidence_threshold:
                    accepted_count += 1
                    if verbose:
                        status = "\033[92m✅ Accepted\033[0m"
                        corrected_str = "-"
                        print(
                            f"│ {i + 1:<3d} │ {draft_token_str:<15.15s} │ {draft_token_prob:>12.2%} │ {status:<26s} │ {corrected_str:<15s} │")

                else:
                    corrected_token = torch.argmax(target_logit, dim=-1).item()
                    if verbose:
                        corrected_str = self.tokenizer.decode(corrected_token).replace('\n', '\\n')
                        status = "\033[91m❌ Rejected\033[0m"
                        print(
                            f"│ {i + 1:<3d} │ {draft_token_str:<15.15s} │ {draft_token_prob:>12.2%} │ {status:<26s} │ {corrected_str:<15.15s} │")
                        print(f"└{'─' * 5}┴{'─' * 17}┴{'─' * 14}┴{'─' * 20}┴{'─' * 17}┘")
                    if accepted_count > 0:
                        generated_token_ids.extend(draft_ids[:accepted_count].tolist())
                    generated_token_ids.append(corrected_token)
                    total_accepted_draft_tokens += accepted_count
                    break
            else:  # All draft tokens accepted
                total_accepted_draft_tokens += accepted_count
                generated_token_ids.extend(draft_ids.tolist())
                last_logit = logits[:, -1, :]
                next_token = torch.argmax(last_logit, dim=-1).item()
                generated_token_ids.append(next_token)
                if verbose:
                    print(f"└{'─' * 5}┴{'─' * 17}┴{'─' * 14}┴{'─' * 20}┴{'─' * 17}┘")
                    accepted_token_strs = [self.tokenizer.decode(t).replace('\n', '\\n') for t in draft_ids.tolist()]
                    next_token_str = self.tokenizer.decode(next_token).replace('\n', '\\n')
                    print(f"✅ \033[92mAccepted all {accepted_count} tokens: \033[0m{accepted_token_strs}")
                    print(f"✅ \033[92mGenerated Target Token: \033[0m{next_token_str}")

            if self.tokenizer.eos_token_id in generated_token_ids:
                eos_index = generated_token_ids.index(self.tokenizer.eos_token_id)
                generated_token_ids = generated_token_ids[:eos_index]
                if verbose: print("🛑 EOS token generated. Stopping.")
                break
            step += 1

        end_time = time.time()
        final_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)

        if verbose:
            total_time = end_time - start_time
            num_generated = len(generated_token_ids)
            acceptance_rate = (total_accepted_draft_tokens / total_draft_tokens) * 100 if total_draft_tokens > 0 else 0
            latency = (total_time / num_generated) * 1000 if num_generated > 0 else float('inf')
            throughput = num_generated / total_time if total_time > 0 else float('inf')

            print("\n" + "\033[95m" + "─" * 50 + "\033[0m")
            print("🏁 Speculative Decoding Finished")
            print(f"\033[94m💬 User Input:\033[0m\n{text}")
            print(f"\n\033[92m🟢 Generated Text:\033[0m\n{final_text}")
            print("\n\033[94m📊 Performance:\033[0m")
            print(f"├─ Total Time: {total_time:.2f}s")
            print(f"├─ Latency: {latency:.2f} ms/token")
            print(f"└─ Throughput: {throughput:.2f} tokens/s")
            print(f"\033[94m✨ Speculative Stats:\033[0m")
            print(f"├─ Acceptance Rate: {acceptance_rate:.2f}%")
            print(f"├─ Total Drafted Tokens: {total_draft_tokens}")
            print(f"└─ Total Accepted Draft Tokens: {total_accepted_draft_tokens}")

        return final_text