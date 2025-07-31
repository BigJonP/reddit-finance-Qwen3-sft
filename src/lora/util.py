import re


def parse_instruction_response(text):
    pattern = r"### Instruction:\s*(.*?)\s*### Response:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)

    instruction = match.group(1).strip()
    response = match.group(2).strip()
    return {"instruction": instruction, "response": response}


def get_model_output(input_text, tokenizer, model):
    messages = [{"role": "user", "content": input_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return content


def escape_markdown_table_cell(text):
    if not text:
        return ""

    text = text.replace("\n", "<br>")

    text = text.replace("|", "\\|")

    text = text.replace("\\", "\\\\")

    if len(text) > 1000:
        text = text[:1000] + "..."

    return text


def log_model_output(
    input_text, response_text, pre_sft_response, post_sft_response, file_path
):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("## Input\n")
        f.write(f"{input_text}\n\n")
        f.write(
            "|                | Ground Truth Response | Pre-SFT Response | Post-SFT Response |\n"
        )
        f.write(
            "|:--------------:|:---------------------:|:----------------:|:-----------------:|\n"
        )

        ground_truth = escape_markdown_table_cell(response_text)
        pre_sft = escape_markdown_table_cell(pre_sft_response)
        post_sft = escape_markdown_table_cell(post_sft_response)

        f.write(f"| **Response**   | {ground_truth} | {pre_sft} | {post_sft} |\n\n")
        f.write("---\n\n")
