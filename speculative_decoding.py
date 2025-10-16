import os
import re
import subprocess
from typing import List, Optional
import torch


def generate_draft_response(
    model_path: str,
    ctx_size: int = 128,
    n_threads: int = 12,
    verbose: bool = True,
    host: str = "127.0.0.1",
    port: int = 8080,
):
    main_path = os.path.join("build", "bin", "llama-server")

    args = [
        main_path,
        "-m", model_path,
        "-c", str(ctx_size),
        "-t", str(n_threads),
        "-ngl", "0",  # CPU
        "-b", "1",  # batch size
        #"-p", prompt,
        #"-n", str(max_new_tokens),
        #"--temp", str(temperature),
        #"--top_p", str(top_p),
        #"--top_k", str(top_k),
        #"--repeat_penalty", str(repeat_penalty),
        #"--seed", str(seed),
        #"--no-warmup",
        '--host', host,
        '--port', str(port),
    ]
    if verbose:
        args.append("-v")

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    stderr = result.stderr

    generated_text = stdout.strip()

    if verbose:
        perf_info = {}
        prompt_match = re.search(
            r"prompt eval time =\s*([\d.]+) ms / *(\d+) tokens .*?([\d.]+) ms per token",
            stderr,
        )
        if prompt_match:
            perf_info["prompt_eval_time_ms"] = float(prompt_match.group(1))
            perf_info["prompt_tokens"] = int(prompt_match.group(2))
            perf_info["prompt_ms_per_token"] = float(prompt_match.group(3))
        eval_match = re.search(
            r"eval time =\s*([\d.]+) ms / *(\d+) runs .*?([\d.]+) ms per token",
            stderr,
        )
        if eval_match:
            perf_info["gen_eval_time_ms"] = float(eval_match.group(1))
            perf_info["gen_tokens"] = int(eval_match.group(2))
            perf_info["gen_ms_per_token"] = float(eval_match.group(3))
        total_match = re.search(
            r"total time =\s*([\d.]+) ms / *(\d+) tokens",
            stderr,
        )
        if total_match:
            perf_info["total_time_ms"] = float(total_match.group(1))
            perf_info["total_tokens"] = int(total_match.group(2))

        unwanted_patterns = ("n_past =", "n_remain:", "eval:")
        filtered_stderr = "\n".join(
            line for line in stderr.splitlines()
            if not line.strip().startswith(unwanted_patterns)
        )

        print("\n\033[95m─────────────────────────────")
        print("⚙️  BitNet Draft Model Generation Report")
        print("─────────────────────────────\033[0m")

        print(f"\033[92mConversation:\n\033[0m {generated_text}\n")

        if perf_info:
            print("\033[93m📊 Performance Metrics:\033[0m")
            for k, v in perf_info.items():
                print(f"  • {k:<25}: {v}")

        print("\n\033[90m─────────────────────────────")
        print("stderr log")
        print("─────────────────────────────\033[0m")
        print(filtered_stderr.strip())
        print("\033[95m─────────────────────────────\033[0m\n")

        return None

    else:
        return generated_text

@torch.no_grad()
def verify_with_target(
    context: str,
    tokenizer,
    target_model,
    num_assistant_tokens: int = 10,
    confidence_threshold: float = 0.4,
    verbose: bool = True
):
    ctx_ids = tokenizer.encode(
        context,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(target_model.device)
    seq_len = ctx_ids.size(1)
    outputs = target_model(
        input_ids=ctx_ids,
        output_hidden_states=False,
        use_cache=False
    )
    logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1)

    tokens_window = ctx_ids[0, seq_len - num_assistant_tokens : seq_len]
    prev_logprobs = logprobs[0, seq_len - num_assistant_tokens - 1 : seq_len - 1, :]

    probs: List[float] = []
    accepted_ids: List[int] = []
    cut_index = num_assistant_tokens

    for i in range(num_assistant_tokens):
        tok_id = tokens_window[i].item()
        prob = prev_logprobs[i, tok_id].exp().item()
        probs.append(prob)
        if prob < confidence_threshold and cut_index == num_assistant_tokens:
            cut_index = i
            break
        else:
            accepted_ids.append(tok_id)

    if verbose:
        rejected_ids = tokens_window[cut_index:].tolist()
        accepted_text = tokenizer.decode(
            accepted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        rejected_text = tokenizer.decode(
            rejected_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print("\n\033[95m─────────────────────────────")
        print("🤖 Target Mode Verification Report")
        print("─────────────────────────────\033[0m")

        print(f"\033[94mDraft Text:\n\033[0m {context}")
        print(f"\033[93mConfidence Threshold:\033[0m {confidence_threshold}\n")

        print("\033[96m🧩 Token Probabilities:\033[0m")
        for i, prob in enumerate(probs):
            tok = tokenizer.decode([tokens_window[i].item()], skip_special_tokens=True)
            status = (
                "\033[92mACCEPTED\033[0m ✅" if prob >= confidence_threshold
                else "\033[91mREJECTED\033[0m ❌"
            )
            print(f"  • Token[{i:02d}]: '{tok or '[SPACE]'}' → P={prob:.3f} → {status}")

        print("\n\033[93m✂️  Cut Index:\033[0m", cut_index)
        print(f"\033[92mAccepted Text:\033[0m {accepted_text or '[EMPTY]'}")
        print(f"\033[91mRejected Text:\033[0m {rejected_text or '[EMPTY]'}")

        print("\033[95m─────────────────────────────\033[0m\n")

        return None

    else:
        return {
            "cut_index": cut_index,
            "probs": probs
        }

@torch.no_grad()
def speculative_decoding(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    target_model,
    max_new_tokens: int = 128,
    num_assistant_tokens: int = 10,
    confidence_threshold: float = 0.4,
    seed: int = 42,
    verbose: bool = True,
    server_url: Optional[str] = None,
):
    draft_sess = None
    if server_url is not None:
        draft_sess = DraftServerSession(server_url)

    accepted_so_far = ""
    total_generated = 0
    draft_calls = 0
    total_proposed_tokens = 0
    total_accepted_tokens = 0
    step_idx = 0

    while total_generated < max_new_tokens:
        prompt = format_falcon_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_response=accepted_so_far
        )

        # Draft generation
        draft_text = draft_sess.draft(
            prompt=prompt,
            accepted_so_far=accepted_so_far,
            n_predict=num_assistant_tokens,
            seed=seed
        )
        draft_ids = tokenizer.encode(
            draft_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(target_model.device)
        draft_len = draft_ids.size(1)
        context = prompt + draft_text
        draft_calls += 1

        # Target verification
        verify_result = verify_with_target(
            context=context,
            tokenizer=tokenizer,
            target_model=target_model,
            num_assistant_tokens=num_assistant_tokens,
            confidence_threshold=confidence_threshold,
            verbose=False
        )
        cut_index = verify_result["cut_index"]
        print(f"cut_index:{cut_index}")

        ctx_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to(target_model.device)
        seq_len = ctx_ids.size(1)
        window_len = min(num_assistant_tokens, draft_len)
        print(f"window_len:{window_len}")
        cut_index = min(cut_index, window_len)
        print(f"cut_index2:{cut_index}")

        total_proposed_tokens += window_len
        total_accepted_tokens += cut_index
        print(f"total_proposed_tokens:{total_proposed_tokens}, total_accepted_tokens:{total_accepted_tokens}")

        start = seq_len - window_len
        accepted_ids = ctx_ids[0, start: start + cut_index]
        accepted_step = tokenizer.decode(
            accepted_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        accepted_so_far += accepted_step
        total_generated += cut_index

        step_idx += 1
        if verbose:
            rate = (cut_index / window_len) if window_len > 0 else 0.0
            print("\n\033[95m─────────────────────────────")
            print(f"🚀 Step {step_idx} — Draft Verification")
            print("─────────────────────────────\033[0m")
            print(f"\033[94mProposed\033[0m: {window_len} | \033[92mAccepted\033[0m: {cut_index} | \033[93mRate\033[0m: {rate:6.2%}")
            if cut_index > 0:
                short_preview = accepted_step.replace("\n", "\\n")
                print(f"\033[92mAccepted Preview:\033[0m {short_preview[:120]}")
                if cut_index == window_len:
                    print(f"\033[96mResult:\033[0m Fully accepted (+{window_len})")
                else:
                    print(f"\033[93mResult:\033[0m Partially accepted (+{cut_index}); continue.")
            else:
                print(f"\033[91mResult:\033[0m Fully rejected (+0)")

        if tokenizer.eos_token in accepted_so_far:
            break
        if total_generated >= max_new_tokens:
            break

    if verbose:
        overall_accept_rate = (total_accepted_tokens / total_proposed_tokens) if total_proposed_tokens > 0 else 0.0

        print("\n\033[95m═════════════════════════════")
        print("📊 Speculative Decoding Summary")
        print("═════════════════════════════\033[0m")
        print(f"• \033[92mGenerated tokens (est)\033[0m : {total_generated}")
        print(f"• \033[94mDraft calls\033[0m             : {draft_calls}")
        print(f"• \033[93mAcceptance rate\033[0m         : {overall_accept_rate:6.2%}")
        print("\n\033[92m=== Generated Text ===\033[0m")
        print(accepted_so_far if accepted_so_far else "[EMPTY]")
        print("\033[95m═════════════════════════════\033[0m\n")

        return {
            "text": accepted_so_far,
            "draft_calls": draft_calls,
            "generated_tokens_est": total_generated,
            "acceptance_rate": overall_accept_rate,
            "accepted_tokens": total_accepted_tokens,
            "proposed_tokens": total_proposed_tokens,
        }
    else:
        return accepted_so_far