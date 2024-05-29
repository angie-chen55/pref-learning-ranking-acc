def maybe_log(logger, msg, level="info"):
    if logger is None:
        return
    logger_fn = getattr(logger, level)
    logger_fn(msg)

# For chat templates not currently implemented in HF
CHAT_USER_TEMPLATE_TULU = "<|user|>\n${user_message}\n"
CHAT_ASSISTANT_TEMPLATE_TULU = "<|assistant|>\n${assistant_message}\n"
CHAT_ASSISTANT_PROMPT_TULU = "<|assistant|>\n"

CHAT_TEMPLATE_MAP = {
    "allenai/tulu-2-dpo-7b": (CHAT_USER_TEMPLATE_TULU, CHAT_ASSISTANT_TEMPLATE_TULU, CHAT_ASSISTANT_PROMPT_TULU),
    "allenai/tulu-2-7b": (CHAT_USER_TEMPLATE_TULU, CHAT_ASSISTANT_TEMPLATE_TULU, CHAT_ASSISTANT_PROMPT_TULU),
}