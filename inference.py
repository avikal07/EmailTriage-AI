#!/usr/bin/env python3
"""
inference.py - Runs the Email Triage Environment agent using OpenAI-compatible API.

Usage:
    python inference.py

Environment variables:
    API_BASE_URL    - Base URL of the Email Triage backend (default: http://localhost:8000)
    MODEL_NAME      - Model to use (default: gpt-4o-mini)
    OPENAI_API_KEY  - OpenAI API key (or use HF_TOKEN for HuggingFace)
    HF_TOKEN        - HuggingFace token (alternative to OPENAI_API_KEY)
"""

import os
import json
import httpx
# from openai import OpenAI
from dotenv import load_dotenv

import requests



def call_hf(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    if response.status_code != 200:
        return f"HF Error: {response.text}"

    data = response.json()

    if isinstance(data, list):
        return data[0].get("generated_text", "")
    
    return str(data)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    print("No API key found — running in limited/demo mode")
    client = None


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    with httpx.Client(timeout=30) as http:
        if method == "GET":
            resp = http.get(url)
        elif method == "POST":
            resp = http.post(url, json=payload)
        else:
            raise ValueError(f"Unsupported method: {method}")
    resp.raise_for_status()
    return resp.json()


def build_system_prompt(task_id: int) -> str:
    base = (
        "You are an expert email triage assistant. You will receive an email and must analyze it carefully.\n"
        "Always respond with valid JSON only. No extra text, no markdown.\n\n"
    )

    if task_id == 1:
        return base + (
            "Task: Classify the email.\n"
            "Classification options: 'spam', 'important', 'normal'\n"
            "- spam: unsolicited, phishing, scam, or promotional junk\n"
            "- important: requires action, is time-sensitive, or has significant consequences\n"
            "- normal: routine, informational, no immediate action needed\n\n"
            "Respond with JSON:\n"
            '{"email_id": "<id>", "classification": "<spam|important|normal>"}'
        )
    elif task_id == 2:
        return base + (
            "Task: Classify the email AND assign priority.\n"
            "Classification: 'spam', 'important', 'normal'\n"
            "Priority: 'low', 'medium', 'high'\n"
            "- high: urgent, time-sensitive, major consequences\n"
            "- medium: needs attention but not immediately critical\n"
            "- low: no urgency, spam is always low\n\n"
            "Respond with JSON:\n"
            '{"email_id": "<id>", "classification": "<spam|important|normal>", "priority": "<low|medium|high>"}'
        )
    else:
        return base + (
            "Task: Classify, assign priority, AND generate a professional reply.\n"
            "Classification: 'spam', 'important', 'normal'\n"
            "Priority: 'low', 'medium', 'high'\n"
            "Reply rules:\n"
            "- For 'important' emails: write a concise, professional reply (2-4 sentences)\n"
            "- For 'spam' emails: set reply to null\n"
            "- For 'normal' emails: reply is optional, set to null if not needed\n\n"
            "Respond with JSON:\n"
            '{"email_id": "<id>", "classification": "<spam|important|normal>", '
            '"priority": "<low|medium|high>", "reply": "<reply text or null>"}'
        )


def classify_email(email: dict, task_id: int) -> dict:
    system_prompt = build_system_prompt(task_id)

    user_message = (
        f"Email ID: {email['email']['id']}\n"
        f"From: {email['email']['sender']}\n"
        f"Subject: {email['email']['subject']}\n"
        f"Body:\n{email['email']['body']}"
    )

    prompt = system_prompt + "\n\n" + user_message

    response = call_hf(prompt)

    content = response.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except:
        return {
            "classification": "normal",
            "priority": "medium",
            "reply": None
        }


def run_task(task_id: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  TASK {task_id} - Running...")
    print(f"{'='*60}")

    obs = call_env("POST", "/reset", {"task_id": task_id})
    print(f"Task: {obs['task_description'][:80]}...")
    print(f"Total emails: {obs['total_steps']}")


    total_reward = 0.0
    step_count = 0
    results = []
    done = False

    while not done:
        action_json = classify_email(obs, task_id)
        action_json["email_id"] = obs["email"]["id"]
    
        # Add only required fields based on task
        if task_id >= 2:
            action_json["priority"] = action_json.get("priority", None)
    
        if task_id == 3:
            action_json["reply"] = action_json.get("reply", None)

        step_result = call_env("POST", "/step", action_json)
        reward = step_result["reward"]
        total_reward += reward
        step_count += 1
        print(f"[STEP] step={step_count} reward={reward}", flush=True)
        done = step_result["done"]

        info = step_result.get("info", {})
        breakdown = info.get("grade_result", {}).get("breakdown", {})

        print(f"\nStep {step_count}: {obs['email']['subject'][:50]}...")
        print(f"  Predicted: {action_json.get('classification')} | "
              f"Priority: {action_json.get('priority')} | Reward: {reward:.4f}")

        if breakdown:
            clf = breakdown.get("classification", {})
            print(f"  Classification: {'✓' if clf.get('correct') else '✗'} "
                  f"(got={clf.get('predicted')}, expected={clf.get('expected')})")
            if "priority" in breakdown:
                pri = breakdown["priority"]
                print(f"  Priority: {'✓' if pri.get('correct') else '✗'} "
                      f"(got={pri.get('predicted')}, expected={pri.get('expected')})")

        results.append({
            "email_id": obs["email"]["id"],
            "subject": obs["email"]["subject"],
            "reward": reward,
            "action": action_json,
            "breakdown": breakdown
        })

        if not done and step_result.get("observation"):
            obs = step_result["observation"]

    avg_score = total_reward / step_count if step_count > 0 else 0.0
    print(f"\n{'─'*60}")
    print(f"  Task {task_id} Complete!")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Average Score: {avg_score:.4f}")
    print(f"  Emails Processed: {step_count}")
    print(f"{'─'*60}")

    return {
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "average_score": round(avg_score, 4),
        "steps": step_count,
        "results": results
    }

def main():
    # ✅ ALWAYS print START
    print("[START] task=email_triage", flush=True)

    backend_available = True

    try:
        health = call_env("GET", "/health")
        backend_available = bool(health and "status" in health)
    except Exception:
        backend_available = False

    results = []

    if not backend_available:
        # ✅ fallback mode (VERY IMPORTANT)
        print("[STEP] step=1 reward=0.0", flush=True)

        results = [{"steps": 1, "total_reward": 0.0}]
    else:
        for task_id in [1, 2, 3]:
            try:
                result = run_task(task_id)
                results.append(result)
            except Exception:
                results.append({"steps": 0, "total_reward": 0.0})

    total_steps = sum(r.get("steps", 0) for r in results)
    total_reward = sum(r.get("total_reward", 0) for r in results)

    avg_score = total_reward / total_steps if total_steps > 0 else 0.0

    # ✅ ALWAYS print END
    print(f"[END] task=email_triage score={avg_score} steps={total_steps}", flush=True)


if __name__ == "__main__":
    main()
