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
    
    # より小さいモデルを使用
    model_name = "microsoft/DialoGPT-small"  # または "distilgpt2" でさらに小さく
    
    # トークナイザーとモデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 8bit量子化でメモリ節約
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 16bit精度でメモリ半減
        device_map="cpu",  # CPU使用
        low_cpu_mem_usage=True  # メモリ使用量削減
    )
    
    # データセットの準備
    dataset = prepare_dataset()
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=True,
            truncation=True,
            max_length=128  # 256から128に減らす
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 訓練データと評価データに分割（評価用に20%使用）
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # トレーニング設定（超軽量版）
    training_args = TrainingArguments(
        output_dir="/app/models/finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,  # エポック数を減らす
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # 勾配累積でメモリ節約
        fp16=True,  # 16bit精度でメモリ半減
        save_steps=100,
        logging_steps=10,
        evaluation_strategy="epoch",  # 評価を無効化してメモリ節約
        save_strategy="epoch",
        max_steps=10,  # 最大ステップ数を制限
    )
    
    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # トレーナーの初期化（評価なし）
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,  # 全データを訓練に使用
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