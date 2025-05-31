import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import os

# Check for required dependencies
try:
    import cv2
except ImportError:
    st.error("❌ OpenCV is not installed. Please install it using: pip install opencv-python-headless")
    st.stop()

try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ Ultralytics is not installed. Please install it using: pip install ultralytics")
    st.stop()

# -------------- Page Config ----------------
st.set_page_config(
    page_title="Traffic AI Vision - YOLOv8 Detection",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="collapsed"
)

# -------------- Enhanced Custom CSS ----------------
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Background and main styling */
    .stApp {
        background-image: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(0,0,0,0.7)), 
                         url("https://www.theseforeignroads.com/wp-content/uploads/2018/09/Essay-Roads-Featured.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
        filter: blur(0.5px);
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(3px);
        z-index: -1;
    }

    /* Main container */
    .main {
        background: linear-gradient(145deg, rgba(0,0,0,0.85), rgba(30,30,50,0.9));
        padding: 2rem 3rem;
        border-radius: 25px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255,255,255,0.1);
        margin: 1rem;
        color: white;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.2);
    }

    .main-title {
        color: white;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.6);
        letter-spacing: -1px;
    }

    .subtitle {
        color: white;
        font-size: 1.8rem;
        font-weight: 400;
        margin: 1rem 0 0 0;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(90deg, rgba(0,0,0,0.7), rgba(30,30,50,0.7));
        padding: 1rem;
        border-radius: 20px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 15px;
        color: white;
        font-weight: 600;
        font-size: 1.4rem;
        padding: 1.2rem 2.5rem;
        transition: all 0.3s ease;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }

    .stTabs [data-baseweb="tab"][data-state="active"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        color: white;
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        transform: translateY(-3px);
        border: 2px solid rgba(255,255,255,0.2);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        font-weight: 700;
        padding: 1.2rem 3rem;
        font-size: 1.3rem;
        border-radius: 20px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 12px 30px rgba(40, 167, 69, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #218838, #1e7e34);
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(40, 167, 69, 0.6);
    }

    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(145deg, rgba(0,0,0,0.6), rgba(30,30,50,0.7));
        border: 3px dashed rgba(102, 126, 234, 0.8);
        border-radius: 25px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        color: white;
    }

    .stFileUploader > div > div:hover {
        border-color: rgba(118, 75, 162, 0.9);
        background: linear-gradient(145deg, rgba(30,30,50,0.8), rgba(0,0,0,0.7));
        transform: scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }

    /* Image containers */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 3px solid transparent;
        background-clip: padding-box;
    }

    .image-container::before {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        z-index: -1;
        margin: -3px;
        border-radius: inherit;
        background: linear-gradient(135deg, #667eea, #764ba2);
    }

    /* Detection results */
    .detection-card {
        background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,50,0.9));
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        border: 2px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        color: white;
    }

    .detection-card h3 {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }

    .detection-item {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.3);
        font-weight: 600;
        font-size: 1.2rem;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        backdrop-filter: blur(5px);
    }

    /* Spinner and alerts */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border-left: 5px solid #28a745;
    }

    .stInfo {
        background: linear-gradient(135deg, #cce7ff, #b3d9ff);
        color: #004085;
        border-left: 5px solid #007bff;
    }

    .stError {
        background: linear-gradient(135deg, #f8d7da, #f1aeb5);
        color: #721c24;
        border-left: 5px solid #dc3545;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(0, 123, 255, 0.3);
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0056b3, #004085);
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 123, 255, 0.4);
    }

    /* Webcam section */
    .webcam-section {
        background: linear-gradient(145deg, rgba(0,0,0,0.7), rgba(30,30,50,0.8));
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255,255,255,0.1);
        color: white;
    }

    .webcam-section h3 {
        color: white;
        font-size: 2rem;
        font-weight: 600;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
    }

    .webcam-section p {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }

    /* Custom divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border: none;
        margin: 2rem 0;
        border-radius: 2px;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 3rem;
        }
        .subtitle {
            font-size: 1.4rem;
        }
        .main {
            padding: 1.5rem;
            margin: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.2rem;
            padding: 1rem 1.5rem;
        }
    }

    /* Global text styling */
    .stMarkdown, .stText, p, div, span, label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        font-weight: 600;
    }

    .stMarkdown h3 {
        font-size: 1.8rem !important;
    }

    .stMarkdown p {
        font-size: 1.2rem !important;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# -------------- Enhanced Header ----------------
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🚦 Traffic AI Vision</h1>
        <p class="subtitle">Advanced YOLOv8 Object Detection for Traffic Analysis</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin: 2rem 0; background: linear-gradient(145deg, rgba(0,0,0,0.7), rgba(30,30,50,0.8)); padding: 2rem; border-radius: 20px; backdrop-filter: blur(10px); border: 2px solid rgba(255,255,255,0.1);">
        <p style="font-size: 1.5rem; color: white; font-weight: 500; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin: 0;">
            🚀 Detect and analyze traffic elements with state-of-the-art AI technology
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# -------------- Load YOLO Model ----------------
@st.cache_resource
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists("best.pt"):
            st.error("❌ Model file 'best.pt' not found. Please ensure the model file is in the project directory.")
            return None
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


model = load_model()

if model is None:
    st.warning("⚠️ Model not loaded. Please check if 'best.pt' file exists in your project directory.")
    st.info("💡 You can download a pre-trained YOLOv8 model from Ultralytics or use your custom trained model.")
    st.stop()

# -------------- Enhanced Tabs ----------------
tab1, tab2 = st.tabs(["📂 Upload & Analyze", "📸 Live Camera Detection"])

# -------------- Enhanced Upload Tab ----------------
with tab1:
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0; background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,50,0.9)); padding: 2rem; border-radius: 20px; backdrop-filter: blur(10px); border: 2px solid rgba(102, 126, 234, 0.3);">
            <h3 style="color: white; font-weight: 600; font-size: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.7);">📁 Upload Your Image</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">Support for JPG, JPEG, and PNG formats</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to detect traffic elements"
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            img = Image.open(uploaded_file)
            st.markdown(
                '<div style="background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,50,0.9)); padding: 1.5rem; border-radius: 20px; backdrop-filter: blur(10px); border: 2px solid rgba(102, 126, 234, 0.3);"><h3 style="color: white; font-size: 1.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin-bottom: 1rem;">🖼️ Original Image</h3></div>',
                unsafe_allow_html=True)
            st.image(img, caption="Input Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file.name)
            image_path = temp_file.name

        with st.spinner("🔍 AI is analyzing your image..."):
            try:
                results = model(image_path)
                results[0].save(filename="output.jpg")

                with col2:
                    st.markdown(
                        '<div style="background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,50,0.9)); padding: 1.5rem; border-radius: 20px; backdrop-filter: blur(10px); border: 2px solid rgba(102, 126, 234, 0.3);"><h3 style="color: white; font-size: 1.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin-bottom: 1rem;">🎯 Detection Results</h3></div>',
                        unsafe_allow_html=True)
                    st.image("output.jpg", caption="AI Analysis", use_column_width=True)

                st.success("✅ Detection completed successfully!")

                # Enhanced results display
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    st.markdown("""
                        <div class="detection-card">
                            <h3 style="color: white; margin-bottom: 1rem;">📊 Detected Objects</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    for i, box in enumerate(results[0].boxes):
                        cls = model.names[int(box.cls)]
                        conf = float(box.conf)
                        st.markdown(f"""
                            <div class="detection-item">
                                <strong>🎯 {cls}</strong> — Confidence: <span style="color: #28a745; font-weight: 600;">{conf:.2%}</span>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No objects detected in this image.")

                # Enhanced download section
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with open("output.jpg", "rb") as f:
                        st.download_button(
                            "📥 Download Analysis Results",
                            f,
                            "traffic_analysis.jpg",
                            "image/jpeg",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")

        # Clean up temporary file
        try:
            os.unlink(image_path)
        except:
            pass

# -------------- Enhanced Webcam Tab ----------------
with tab2:
    st.markdown("""
        <div class="webcam-section">
            <h3 style="color: white; margin-bottom: 1rem;">📹 Real-time Camera Detection</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 2rem;">Capture images directly from your camera for instant analysis</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        cap_btn = st.button("🎥 Capture & Analyze", use_container_width=True)

    if cap_btn:
        with st.spinner("📸 Accessing camera and capturing image..."):
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error(
                        "❌ Could not access webcam. Please check your camera permissions and ensure no other application is using the camera.")
                else:
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        frame = cv2.resize(frame, (640, 480))
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        img_pil.save("webcam.jpg")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                '<div style="background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,50,0.9)); padding: 1.5rem; border-radius: 20px; backdrop-filter: blur(10px); border: 2px solid rgba(102, 126, 234, 0.3);"><h3 style="color: white; font-size: 1.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin-bottom: 1rem;">📸 Captured Image</h3></div>',
                                unsafe_allow_html=True)
                            st.image(img_pil, caption="Live Capture", use_column_width=True)

                        with st.spinner("🔍 Analyzing captured image..."):
                            results = model("webcam.jpg")
                            results[0].save(filename="webcam_output.jpg")

                            with col2:
                                st.markdown(
                                    '<div style="background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,50,0.9)); padding: 1.5rem; border-radius: 20px; backdrop-filter: blur(10px); border: 2px solid rgba(102, 126, 234, 0.3);"><h3 style="color: white; font-size: 1.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); margin-bottom: 1rem;">🎯 Live Analysis</h3></div>',
                                    unsafe_allow_html=True)
                                st.image("webcam_output.jpg", caption="AI Detection", use_column_width=True)

                        st.success("✅ Live detection completed!")

                        # Results display
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            st.markdown("""
                                <div class="detection-card">
                                    <h3 style="color: white; margin-bottom: 1rem;">📊 Live Detection Results</h3>
                                </div>
                            """, unsafe_allow_html=True)

                            for box in results[0].boxes:
                                cls = model.names[int(box.cls)]
                                conf = float(box.conf)
                                st.markdown(f"""
                                    <div class="detection-item">
                                        <strong>🎯 {cls}</strong> — Confidence: <span style="color: #28a745; font-weight: 600;">{conf:.2%}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("ℹ️ No objects detected in the captured image.")

                        # Download option
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            with open("webcam_output.jpg", "rb") as f:
                                st.download_button(
                                    "📥 Download Live Analysis",
                                    f,
                                    "live_traffic_analysis.jpg",
                                    "image/jpeg",
                                    use_container_width=True
                                )
                    else:
                        st.error("❌ Could not capture image from webcam.")

            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")

# -------------- Footer ----------------
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #6c757d;">
        <p style="margin: 0; font-size: 0.9rem;">
            🚦 Traffic AI Vision | Powered by YOLOv8 & Streamlit | 
            <span style="color: #667eea;">Advanced Computer Vision Technology</span>
        </p>
    </div>
""", unsafe_allow_html=True)