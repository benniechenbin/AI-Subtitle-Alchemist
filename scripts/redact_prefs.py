import json
import os


def redact_preferences(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 脱敏敏感字段
        sensitive_keys = ["llm_key", "api_key", "deepseek_api_key", "openai_api_key"]

        redacted_count = 0
        for key in sensitive_keys:
            if data.get(key):
                data[key] = "***REDACTED***"
                redacted_count += 1

        if redacted_count > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Successfully redacted {redacted_count} fields in {file_path}")
        else:
            print("No sensitive fields found or already redacted.")

    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")


if __name__ == "__main__":
    # 默认处理当前目录下的 user_preferences.json
    redact_preferences("user_preferences.json")
