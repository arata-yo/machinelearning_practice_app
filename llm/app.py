from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
from typing import Dict, List

app = Flask(__name__)
CORS(app)  # CORS有効化

# グローバル変数
model = None
tokenizer = None
qa_pipeline = None
qa_data = {}

def load_qa_data():
    """Q&Aデータを読み込む"""
    global qa_data
    try:
        with open('training_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                qa_data[item['instruction'].lower()] = item['output']
        print(f"Q&Aデータを{len(qa_data)}件読み込みました")
    except Exception as e:
        print(f"Q&Aデータ読み込みエラー: {e}")

def load_model():
    """モデルを読み込む（簡易版：ルールベース＋小規模LLM）"""
    global model, tokenizer, qa_pipeline
    
    # まずはルールベースのQ&Aを使用
    load_qa_data()
    
    # 軽量モデルの読み込み（オプション）
    try:
        # メモリ節約のため、最初はルールベースのみ使用
        # 必要に応じて小規模モデルを読み込む
        model_name = "microsoft/DialoGPT-small"  # 軽量な対話モデル
        
        # モデルが存在しない場合はスキップ
        if os.path.exists('/app/models/dialogpt'):
            tokenizer = AutoTokenizer.from_pretrained('/app/models/dialogpt')
            model = AutoModelForCausalLM.from_pretrained('/app/models/dialogpt')
            print("DialogGPTモデルを読み込みました")
        else:
            print("ルールベースモードで動作します")
            
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        print("ルールベースモードで動作します")

def find_best_answer(question: str) -> Dict:
    """質問に最も適した回答を見つける"""
    question_lower = question.lower().strip()
    
    # 完全一致を探す
    if question_lower in qa_data:
        return {
            'answer': qa_data[question_lower],
            'confidence': 1.0,
            'source': 'exact_match'
        }
    
    # 部分一致を探す
    best_match = None
    best_score = 0
    
    for q, a in qa_data.items():
        # キーワードベースのマッチング
        keywords_in_db = set(q.split())
        keywords_in_question = set(question_lower.split())
        
        # 共通キーワードの数を数える
        common_keywords = keywords_in_db & keywords_in_question
        if len(common_keywords) > best_score:
            best_score = len(common_keywords)
            best_match = a
    
    if best_match and best_score > 0:
        confidence = min(best_score * 0.3, 0.9)  # 最大0.9の信頼度
        return {
            'answer': best_match,
            'confidence': confidence,
            'source': 'keyword_match'
        }
    
    # デフォルトの回答
    return {
        'answer': '申し訳ございません。その質問についての情報が見つかりませんでした。アヤメの分類、入力値、予測結果について質問してください。',
        'confidence': 0.1,
        'source': 'default'
    }

@app.route('/chat', methods=['POST'])
def chat():
    """チャットエンドポイント"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'メッセージが空です'}), 400
        
        # 回答を検索
        result = find_best_answer(message)
        
        return jsonify({
            'response': result['answer'],
            'confidence': result['confidence'],
            'source': result['source']
        })
        
    except Exception as e:
        app.logger.error(f"チャットエラー: {e}")
        return jsonify({
            'error': 'エラーが発生しました',
            'response': '申し訳ございません。エラーが発生しました。'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'qa_data_loaded': len(qa_data) > 0,
        'model_loaded': model is not None
    })

@app.route('/qa_list', methods=['GET'])
def qa_list():
    """利用可能な質問のリストを返す"""
    questions = list(qa_data.keys())
    return jsonify({
        'questions': questions[:10],  # 最初の10個を返す
        'total': len(questions)
    })

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=True)