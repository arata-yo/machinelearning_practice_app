from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# データベース設定
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'postgresql://user:password@postgres:5432/mlapp'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# モデルとモデル情報の読み込み
MODEL_PATH = '/app/models/iris_lgb_model.pkl'
MODEL_INFO_PATH = '/app/models/model_info.pkl'

# グローバル変数でモデルを保持
model = None
model_info = None

def load_model():
    """モデルを読み込む関数"""
    global model, model_info
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"モデルを読み込みました: {MODEL_PATH}")
        else:
            print(f"警告: モデルファイルが見つかりません: {MODEL_PATH}")
            
        if os.path.exists(MODEL_INFO_PATH):
            model_info = joblib.load(MODEL_INFO_PATH)
            print(f"モデル情報を読み込みました: {MODEL_INFO_PATH}")
        else:
            # デフォルトのモデル情報
            model_info = {
                'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                'target_names': ['setosa', 'versicolor', 'virginica']
            }
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")

# データベースモデル
class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    sepal_length = db.Column(db.Float, nullable=False)
    sepal_width = db.Column(db.Float, nullable=False)
    petal_length = db.Column(db.Float, nullable=False)
    petal_width = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    prediction_name = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'sepal_length': self.sepal_length,
            'sepal_width': self.sepal_width,
            'petal_length': self.petal_length,
            'petal_width': self.petal_width,
            'prediction': self.prediction,
            'prediction_name': self.prediction_name,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ルート
@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """予測API"""
    try:
        # リクエストデータの取得
        data = request.json
        
        # 入力値の検証
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field}が必要です'}), 400
        
        # 数値に変換
        try:
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])
        except ValueError:
            return jsonify({'error': '入力値は数値である必要があります'}), 400
        
        # 値の範囲チェック
        if not (0 < sepal_length < 20 and 0 < sepal_width < 20 and 
                0 < petal_length < 20 and 0 < petal_width < 20):
            return jsonify({'error': '入力値が有効な範囲外です'}), 400
        
        # モデルが読み込まれていない場合
        if model is None:
            # ダミーの予測を返す（開発用）
            import random
            prediction = random.randint(0, 2)
            confidence = random.uniform(0.7, 0.99)
            prediction_name = model_info['target_names'][prediction]
        else:
            # 予測の実行
            input_data = pd.DataFrame({
                'sepal_length': [sepal_length],
                'sepal_width': [sepal_width],
                'petal_length': [petal_length],
                'petal_width': [petal_width]
            })
            
            pred_proba = model.predict(input_data, num_iteration=model.best_iteration)
            prediction = int(np.argmax(pred_proba, axis=1)[0])
            confidence = float(np.max(pred_proba))
            prediction_name = model_info['target_names'][prediction]
            
            # 各クラスの確率
            probabilities = {
                model_info['target_names'][i]: float(prob) 
                for i, prob in enumerate(pred_proba[0])
            }
        
        # データベースに保存
        pred_record = Prediction(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            prediction=prediction,
            prediction_name=prediction_name,
            confidence=confidence
        )
        db.session.add(pred_record)
        db.session.commit()
        
        # 結果を返す
        result = {
            'prediction': prediction,
            'prediction_name': prediction_name,
            'confidence': confidence,
            'message': get_result_message(prediction_name, confidence)
        }
        
        if model is not None:
            result['probabilities'] = probabilities
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"予測エラー: {e}")
        return jsonify({'error': '予測中にエラーが発生しました'}), 500

@app.route('/history')
def history():
    """予測履歴ページ"""
    try:
        # 最新の予測を取得
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        app.logger.error(f"履歴取得エラー: {e}")
        return render_template('history.html', predictions=[])

@app.route('/api/history')
def api_history():
    """予測履歴API"""
    try:
        limit = request.args.get('limit', 20, type=int)
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(limit).all()
        return jsonify([p.to_dict() for p in predictions])
    except Exception as e:
        app.logger.error(f"履歴API エラー: {e}")
        return jsonify({'error': '履歴の取得に失敗しました'}), 500

@app.route('/health')
def health():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database': 'connected'
    })

def get_result_message(prediction_name, confidence):
    """予測結果に応じたメッセージを生成"""
    confidence_level = "非常に高い" if confidence > 0.9 else "高い" if confidence > 0.7 else "中程度の"
    
    messages = {
        'setosa': f"この花は{confidence_level}確率で「セトサ（Iris setosa）」です。セトサは最も小さく、がく片が幅広いのが特徴です。",
        'versicolor': f"この花は{confidence_level}確率で「バーシカラー（Iris versicolor）」です。バーシカラーは中間的なサイズで、青紫色の花を咲かせます。",
        'virginica': f"この花は{confidence_level}確率で「バージニカ（Iris virginica）」です。バージニカは最も大きく、花弁が長いのが特徴です。"
    }
    
    return messages.get(prediction_name, f"この花は{prediction_name}の可能性が{confidence_level}です。")

@app.before_request
def before_request():
    """リクエスト前の処理"""
    # モデルが読み込まれていない場合は読み込む
    if model is None:
        load_model()

if __name__ == '__main__':
    with app.app_context():
        # テーブルの作成
        db.create_all()
        # モデルの読み込み
        load_model()
    
    app.run(host='0.0.0.0', port=5000, debug=True)