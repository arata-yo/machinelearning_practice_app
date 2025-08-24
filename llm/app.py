from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from typing import Dict, List
import logging

# ログ設定を最初に行う（DEBUGレベルに変更）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # CORS有効化

# グローバル変数
model = None
tokenizer = None
qa_pipeline = None
qa_data = {}
use_model = False  # モデル使用フラグを追加

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

def debug_environment():
    """環境のデバッグ情報を出力"""
    print("=" * 60)
    print("環境デバッグ情報")
    print("=" * 60)
    
    # カレントディレクトリ
    print(f"カレントディレクトリ: {os.getcwd()}")
    print(f"カレントディレクトリ内容: {os.listdir('.')}")
    
    # ルートディレクトリ
    print(f"\nルートディレクトリ内容: {os.listdir('/')}")
    
    # /appディレクトリ
    if os.path.exists('/app'):
        print(f"\n/appディレクトリ内容: {os.listdir('/app')}")
        
        # iris_modelディレクトリの詳細確認
        model_path = '/app/iris_model'
        if os.path.exists(model_path):
            print(f"\n{model_path}が存在します")
            print(f"{model_path}はディレクトリ: {os.path.isdir(model_path)}")
            print(f"{model_path}の内容: {os.listdir(model_path)}")
            
            # 各ファイルのサイズも確認
            for file in os.listdir(model_path):
                file_path = os.path.join(model_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  - {file}: {size:,} bytes")
        else:
            print(f"\n{model_path}が存在しません")
    else:
        print("\n/appディレクトリが存在しません")
    
    print("=" * 60)

def load_model():
    """モデルを読み込む（簡易版：ルールベース＋小規模LLM）"""
    global model, tokenizer, qa_pipeline, use_model
    
    # まずはルールベースのQ&Aを読み込む
    load_qa_data()
    
    # 環境デバッグ情報を出力
    debug_environment()
    
    # モデル読み込み処理
    model_path = '/app/iris_model'
    
    try:
        print("\n" + "=" * 60)
        print("モデル読み込み処理開始")
        print("=" * 60)
        
        # ディレクトリの存在確認
        if not os.path.exists(model_path):
            print(f"エラー: {model_path} が存在しません")
            print("ルールベースモードで動作します")
            use_model = False
            return
        
        if not os.path.isdir(model_path):
            print(f"エラー: {model_path} はディレクトリではありません")
            print("ルールベースモードで動作します")
            use_model = False
            return
        
        # transformersライブラリのインポート
        print("\ntransformersライブラリをインポート中...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
            import torch
            print("インポート成功")
            print(f"PyTorchバージョン: {torch.__version__}")
        except ImportError as e:
            print(f"transformersまたはtorchのインポートに失敗: {e}")
            print("必要なライブラリがインストールされていません")
            print("ルールベースモードで動作します")
            use_model = False
            return
        
        # config.jsonを読み込んでモデルタイプを確認
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
            model_type = config.get('model_type', 'gpt2')
            print(f"\nモデルタイプ: {model_type}")
        
        # モデルとトークナイザーの読み込み
        print(f"\nモデルを読み込み中: {model_path}")
        
        # トークナイザーの読み込み（tokenizer.jsonの問題を回避）
        print("ステップ1: トークナイザーを読み込み中...")
        
        # vocab.jsonとmerges.txtのパス
        vocab_file = os.path.join(model_path, 'vocab.json')
        merges_file = os.path.join(model_path, 'merges.txt')
        
        print(f"  vocab.json: {os.path.exists(vocab_file)}")
        print(f"  merges.txt: {os.path.exists(merges_file)}")
        
        tokenizer = None
        
        # 方法1: vocab.jsonとmerges.txtから直接読み込み（tokenizer.jsonを回避）
        if os.path.exists(vocab_file) and os.path.exists(merges_file):
            try:
                print("  方法1: vocab.jsonとmerges.txtから直接読み込み中...")
                tokenizer = GPT2Tokenizer(
                    vocab_file=vocab_file,
                    merges_file=merges_file
                )
                # 特殊トークンの設定
                tokenizer.pad_token = tokenizer.eos_token
                print("  ✓ トークナイザーの読み込み成功")
                print(f"  語彙サイズ: {len(tokenizer.get_vocab())}")
            except Exception as e:
                print(f"  ✗ 直接読み込み失敗: {e}")
                tokenizer = None
        
        # 方法2: use_fast=Falseで読み込み（tokenizer.jsonを使わない）
        if tokenizer is None:
            try:
                print("  方法2: use_fast=Falseで読み込み中...")
                # TOKENIZERS_PARALLELISMを無効化してエラーを回避
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    use_fast=False,
                    local_files_only=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print("  ✓ レガシートークナイザーの読み込み成功")
            except Exception as e:
                print(f"  ✗ レガシートークナイザー失敗: {e}")
                tokenizer = None
        
        # 方法3: GPT2Tokenizerを明示的に使用
        if tokenizer is None:
            try:
                print("  方法3: GPT2Tokenizerクラスを直接使用...")
                # tokenizer_config.jsonを一時的に無視して読み込み
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained(
                    model_path,
                    use_fast=False,
                    local_files_only=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print("  ✓ GPT2Tokenizerの読み込み成功")
            except Exception as e:
                print(f"  ✗ GPT2Tokenizer失敗: {e}")
                tokenizer = None
        
        # 最終手段: デフォルトのGPT2トークナイザー
        if tokenizer is None:
            print("  最終手段: デフォルトのGPT2トークナイザーを使用...")
            print("  警告: カスタムトークナイザーではなく、標準のGPT2トークナイザーを使用します")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            print("  ✓ デフォルトGPT2トークナイザーの読み込み成功")
        
        # パディングトークンの設定（必要な場合）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("パディングトークンを設定しました")
        
        # モデルの読み込み
        print("\nステップ2: モデルを読み込み中...")
        
        # model.safetensorsファイルが存在する場合の対応
        model_file = os.path.join(model_path, 'model.safetensors')
        if os.path.exists(model_file):
            print(f"  model.safetensorsファイルを検出 (サイズ: {os.path.getsize(model_file):,} bytes)")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True  # カスタムコードを信頼
            )
            print("  ✓ モデルの読み込み成功")
        except Exception as e:
            print(f"  ✗ モデル読み込みエラー: {e}")
            # safetensorsの問題の可能性がある場合
            if "safetensors" in str(e).lower():
                print("\n  safetensorsの読み込みに問題がある可能性があります。")
                print("  pytorch_model.binファイルへの変換を試みます...")
                
                # ここで変換を試みる（オプション）
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                    # 一時的にstate_dictから読み込む
                    from transformers import GPT2LMHeadModel
                    model = GPT2LMHeadModel(config=config)
                    model.load_state_dict(state_dict)
                    print("  ✓ safetensorsからの直接読み込み成功")
                except Exception as conv_error:
                    print(f"  ✗ safetensors変換エラー: {conv_error}")
                    raise e
            else:
                raise e
        
        # モデルを評価モードに設定
        model.eval()
        print("モデルを評価モードに設定しました")
        
        # モデルを強制的に有効化
        use_model = True
        print("\n" + "=" * 60)
        print("✅ ファインチューニング済みモデルで動作します")
        print("⚠️ 注意: 応答品質は改善の余地があります")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nモデル読み込みエラー: {e}")
        print("\n詳細なエラー情報:")
        import traceback
        print(traceback.format_exc())
        print("\n" + "=" * 60)
        print("⚠️ ルールベースモードで動作します")
        print("=" * 60)
        use_model = False

def generate_model_response(message: str) -> str:
    """モデルを使用して応答を生成"""
    global model, tokenizer
    
    try:
        # 入力をトークナイズ
        inputs = tokenizer.encode(
            message,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # デコード
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 入力部分を除去（必要に応じて）
        if response.startswith(message):
            response = response[len(message):].strip()
        
        return response
        
    except Exception as e:
        print(f"モデル推論エラー: {e}")
        return None

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
        
        response_data = {}
        
        # モデルが利用可能な場合は優先的に使用
        if use_model and model and tokenizer:
            print(f"モデルを使用して応答を生成: {message}")
            model_response = generate_model_response(message)
            
            if model_response:
                response_data = {
                    'response': model_response,
                    'confidence': 0.95,
                    'source': 'finetuned_model'
                }
            else:
                # モデル応答が失敗した場合はルールベースにフォールバック
                result = find_best_answer(message)
                response_data = result
        else:
            # ルールベースの回答を使用
            result = find_best_answer(message)
            response_data = result
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        app.logger.error(f"チャットエラー: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'response': '申し訳ございません。エラーが発生しました。'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'qa_data_loaded': len(qa_data) > 0,
        'model_loaded': model is not None,
        'use_model': use_model,
        'mode': 'finetuned_model' if use_model else 'rule_based'
    })

@app.route('/qa_list', methods=['GET'])
def qa_list():
    """利用可能な質問のリストを返す"""
    questions = list(qa_data.keys())
    return jsonify({
        'questions': questions[:10],  # 最初の10個を返す
        'total': len(questions)
    })

@app.route('/debug', methods=['GET'])
def debug():
    """デバッグ情報を返す"""
    debug_info = {
        'current_dir': os.getcwd(),
        'current_dir_files': os.listdir('.'),
        'app_dir_exists': os.path.exists('/app'),
        'model_dir_exists': os.path.exists('/app/iris_model'),
        'use_model': use_model,
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'qa_data_count': len(qa_data)
    }
    
    if os.path.exists('/app'):
        debug_info['app_dir_files'] = os.listdir('/app')
    
    if os.path.exists('/app/iris_model'):
        debug_info['model_dir_files'] = os.listdir('/app/iris_model')
    
    return jsonify(debug_info)

if __name__ == '__main__':
    # torchのインポートチェック
    try:
        import torch
        print(f"PyTorch利用可能: バージョン {torch.__version__}")
    except ImportError:
        print("警告: PyTorchがインストールされていません")
    
    # transformersのインポートチェック  
    try:
        import transformers
        print(f"Transformers利用可能: バージョン {transformers.__version__}")
    except ImportError:
        print("警告: Transformersがインストールされていません")
    
    # モデルを読み込む
    load_model()
    
    # Flaskアプリを起動
    app.run(host='0.0.0.0', port=5001, debug=True)