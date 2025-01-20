import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# 페이지 설정 (가장 첫 번째 줄에 위치)
st.set_page_config(layout="wide")

# TensorFlow 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# DepthwiseConv2D 오류 해결을 위한 사용자 정의 객체
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# 사용자 정의 객체를 등록
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# CSS 스타일을 사용해 st.progress의 색상 변경
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #FF5733, #FFBD33);  /* 새로운 그라데이션 색상 */
    }
    </style>
    """, unsafe_allow_html=True)

def load_classification_model():
    model = load_model("keras_Model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    # UTF-8 인코딩으로 라벨 파일 읽기
    try:
        class_names = [line.strip()[2:] for line in open("labels.txt", "r", encoding='utf-8').readlines()]
    except UnicodeDecodeError:
        try:
            class_names = [line.strip()[2:] for line in open("labels.txt", "r", encoding='cp949').readlines()]
        except UnicodeDecodeError:
            class_names = [line.strip()[2:] for line in open("labels.txt", "r", encoding='euc-kr').readlines()]
    return model, class_names

def process_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def get_predictions(model, data, class_names):
    predictions = model.predict(data)[0]
    results = [{"class": class_name, "probability": float(prob)} for class_name, prob in zip(class_names, predictions)]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

# 사이드바 설정
with st.sidebar:
    st.title("📋 사용 가이드")
    st.markdown("""
    ### 사용 방법
    1. 오른쪽 메인 화면에서 '이미지 업로드' 버튼을 클릭합니다.
    2. 분류하고 싶은 이미지를 선택합니다.
    3. 자동으로 이미지 분석이 실행됩니다.
    
    ### 분석 결과 해석
    - 각 클래스별 인식 확률이 막대 그래프로 표시됩니다
    - 가장 높은 확률의 클래스가 최종 예측 결과입니다
    
    ### 주의사항
    - 지원되는 이미지 형식: JPG, PNG
    - 이미지는 자동으로 224x224 크기로 조정됩니다.
    """)

# 메인 화면
st.title("🔍 이미지 분류 시스템")

# 파일 업로더 생성
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

# 이미지가 업로드되면 처리 시작
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("업로드된 이미지")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("분류 결과")
        
        try:
            # 모델 로드
            model, class_names = load_classification_model()
            # 이미지 처리
            processed_data = process_image(image)
            # 모든 클래스에 대한 예측 수행
            results = get_predictions(model, processed_data, class_names)
            
            # 최상위 예측 결과 강조 표시
            st.success(f"가장 높은 확률의 클래스: {results[0]['class']} ({results[0]['probability']*100:.2f}%)")
            
            # st.progress로 클래스별 확률 표시
            for result in results:
                class_name = result["class"]
                confidence_score = result["probability"]
                st.write(f"**{class_name}**: {confidence_score*100:.2f}%")
                st.progress(confidence_score)
                st.write("---")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
