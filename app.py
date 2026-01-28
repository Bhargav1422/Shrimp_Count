import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="🦐 Shrimp Counter", layout="centered")
st.title("🦐 Shrimp Counter")

#---------------------------
#Sidebar Parameters
#---------------------------
st.sidebar.header("Settings")
PATCH_SIZE = st.sidebar.slider("Patch Size (px)", 50, 200, 110)
NUM_PATCHES = st.sidebar.slider("Number of Random Patches", 5, 50, 25)
MIN_CONTOUR_AREA = st.sidebar.slider("Min Shrimp Contour Area", 5, 50, 15)
CALIBRATION_FACTOR = st.sidebar.slider("Calibration Factor", 0.5, 1.2, 0.91, 0.01)

#---------------------------
#1. IMAGE UPLOAD
#---------------------------
uploaded_file = st.file_uploader("Upload a bowl image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    orig = img_np.copy()
    h, w = img_np.shape[:2]

    # ---------------------------
    # 2. DETECT BOWL
    # ---------------------------
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=int(min(h,w)*0.35),
        maxRadius=int(min(h,w)*0.48)
    )

    if circles is None:
        st.error("❌ Bowl not detected. Try a clearer image.")
    else:
        cx, cy, radius = np.uint16(np.around(circles))[0][0]

        # Bowl mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius-10, 255, -1)

        # ---------------------------
        # 3. RANDOM PATCH SELECTION
        # ---------------------------
        patches, patch_coords = [], []
        attempts = 0
        while len(patches) < NUM_PATCHES and attempts < 5000:
            attempts += 1
            x = random.randint(0, w-PATCH_SIZE)
            y = random.randint(0, h-PATCH_SIZE)
            px, py = x+PATCH_SIZE//2, y+PATCH_SIZE//2
            if mask[py, px] == 255:
                patches.append(img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
                patch_coords.append((x,y))

        # ---------------------------
        # 4. SHRIMP COUNT FUNCTION
        # ---------------------------
        def count_shrimps_patch(patch):
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            gray_patch = clahe.apply(gray_patch)
            blur_patch = cv2.GaussianBlur(gray_patch, (3,3), 0)
            thresh = cv2.adaptiveThreshold(
                blur_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            contours,  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len([c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA])

        # ---------------------------
        # 5. COUNT ALL PATCHES
        # ---------------------------
        counts = [count_shrimps_patch(p) for p in patches]
        avg_per_patch = sum(counts) / len(counts)
        total_area = np.count_nonzero(mask)
        patch_area = PATCH_SIZE * PATCH_SIZE
        estimated_total = int((total_area / patch_area) * avg_per_patch * CALIBRATION_FACTOR)

        # ---------------------------
        # 6. DISPLAY RESULTS
        # ---------------------------
        st.success(f"*Estimated Shrimp Count:* {estimated_total}")
        st.write(f"Average per patch: {avg_per_patch:.2f}")
        st.write(f"Patches used: {len(patches)}")

        # ---------------------------
        # 7. VISUAL CONFIRMATION
        # ---------------------------
        vis = orig.copy()
        cv2.circle(vis, (cx, cy), radius, (255,0,0), 2)
        for x, y in patch_coords:
            cv2.rectangle(vis, (x,y), (x+PATCH_SIZE, y+PATCH_SIZE), (0,255,0), 2)

        st.image(vis, channels="BGR", caption="Bowl and sampled patches")
