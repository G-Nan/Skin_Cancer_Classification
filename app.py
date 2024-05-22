import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from transformers import ViTForImageClassification, ViTConfig
from safetensors.torch import load_file
from PIL import Image
from torchvision import transforms

# Flask 앱 초기화
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 모델 설정
config_path = 'config.json'
model_path = 'model.safetensors'
config = ViTConfig.from_json_file(config_path)
state_dict = load_file(model_path)
model = ViTForImageClassification(config)
model.load_state_dict(state_dict)
model.eval()

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.CenterCrop(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 라벨 변환 (키를 문자열로 변환)
id2label = {str(key): value for key, value in config.id2label.items()}

# 루트 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 예측
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        # 이미지 예측
        image = Image.open(file_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_label = id2label[str(predicted_class_idx)]
        return render_template('result.html', label=predicted_class_label, file_path=file_path)

# 결과 페이지
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
