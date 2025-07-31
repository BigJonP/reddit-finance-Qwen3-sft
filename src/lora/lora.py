from constants import MODEL_NAME, SAMPLES
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from util import get_model_output, log_model_output, parse_instruction_response


dataset = load_dataset("json", data_files="src/lora/training_data.jsonl", split="train")
sample_rows = [parse_instruction_response(dataset[sample]["text"]) for sample in SAMPLES]
dataset = dataset.train_test_split(test_size=0.1)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pre_sft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

sft_model = get_peft_model(pre_sft_model, lora_config)

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir=f"./{MODEL_NAME}-sft",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    metric_for_best_model="loss",
    fp16=True,
)

trainer = Trainer(
    model=sft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

sft_model.train()

for sample_row in sample_rows:
    pre_output = get_model_output(sample_row["instruction"], tokenizer, pre_sft_model)
    post_output = get_model_output(sample_row["instruction"], tokenizer, sft_model)
    log_model_output(
        sample_row["instruction"],
        sample_row["response"],
        pre_output,
        post_output,
        "results.md",
    )

sft_model.save_pretrained(f"./{MODEL_NAME}-model-checkpoint")
tokenizer.save_pretrained(f"./{MODEL_NAME}-tokenizer-checkpoint")
