"""
簡易ファインチューニングスクリプト
実行: python fine_tune.py
"""
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

def prepare_dataset():
    """データセットの準備"""
    data = []
    with open('training_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # プロンプトと回答を結合
            text = f"質問: {item['instruction']}\n回答: {item['output']}"
            data.append({'text': text})
    
    return Dataset.from_list(data)

def fine_tune_model():
    """モデルのファインチューニング"""
    print("ファインチューニングを開始します...")
    
    # 小規模モデルを使用（メモリ節約）
    model_name = "microsoft/DialoGPT-small"
    
    # トークナイザーとモデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # データセットの準備
    dataset = prepare_dataset()
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=True,
            truncation=True,
            max_length=256
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 訓練データと評価データに分割（評価用に20%使用）
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # トレーニング設定（軽量版）
    training_args = TrainingArguments(
        output_dir="/app/models/finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=50,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=10,
        warmup_steps=10,
        logging_dir='/app/logs',
        # 以下の3つを一致させる
        evaluation_strategy="steps",  # stepsに統一
        save_strategy="steps",         # stepsに統一
        eval_steps=50,                 # save_stepsと同じ値
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    
    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 評価データセットを追加
    )
    
    # トレーニング実行
    trainer.train()
    
    # モデルの保存
    trainer.save_model("/app/models/finetuned")
    tokenizer.save_pretrained("/app/models/finetuned")
    
    print("ファインチューニング完了！")

if __name__ == "__main__":
    # メモリが十分な場合のみ実行
    try:
        fine_tune_model()
    except Exception as e:
        print(f"ファインチューニングエラー: {e}")
        print("ルールベースモードを使用してください")