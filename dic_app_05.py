# 言われた通り，作ってみた
# マーカーが多数検出される＞注目エリアを指定するように変更
# 変形前後で別々にROIを指定できるように変更
# それでも不十分。ユーザー4点選択から微調整に変更
# 使い勝手を考えてさらに修正

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import hashlib

def file_hash(file):
    file.seek(0)
    h = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return h

st.title("DICマーカー：ROI・マーカパラメータ維持徹底仕様")

files = st.file_uploader("旧画像→新画像（2枚アップ）", type=['jpg','png'], accept_multiple_files=True, key="fileuploader")
if len(files) != 2:
    st.info("2枚アップしてください")
    st.stop()
file_key_old = file_hash(files[0])
file_key_new = file_hash(files[1])

def to_gray(file): return np.array(Image.open(file).convert('L'))
img0, img1 = to_gray(files[0]), to_gray(files[1])
h0, w0 = img0.shape
h1, w1 = img1.shape

# ROI/マーカ情報の維持用（すべてセッション管理）
def set_default_roi(w, h):
    return {
        "x_left": int(w * 0.25),
        "x_right": int(w * 0.75),
        "y_top": int(h * 0.25),
        "y_bottom": int(h * 0.75)
    }
if "roi_dict" not in st.session_state:
    st.session_state["roi_dict"] = {}
roi_dict = st.session_state["roi_dict"]

if "last_used_new_roi" not in st.session_state:
    st.session_state["last_used_new_roi"] = set_default_roi(w1, h1)

if "pts_dict" not in st.session_state:
    st.session_state["pts_dict"] = {}

if "skip_flag_dict" not in st.session_state:
    st.session_state["skip_flag_dict"] = {}

# ★マーカ自動候補パラメータもセッション管理（2枚目入替時も維持！）
if "marker_auto_params" not in st.session_state:
    st.session_state["marker_auto_params"] = {"thresh": 90, "area_min": 50, "area_max": 200}

# 旧画像ファイル変更で旧ROI・旧マーカ以外をリセット
def check_and_reset_if_img_changed(side, file_key, w, h):
    hash_session_key = f"last_file_{side}"
    if st.session_state.get(hash_session_key, None) != file_key:
        if side == "new":
            # 新画像入替: ROI維持（last_used_new_roi利用）、マーカ消去のみ
            roi_dict[file_key] = st.session_state["last_used_new_roi"].copy()
        else:
            roi_dict[file_key] = set_default_roi(w, h)
        pts_key = f"{side}_pts_{file_key}"
        st.session_state["pts_dict"][pts_key] = []
        st.session_state[f"skip_flag_{side}"] = False
    st.session_state[hash_session_key] = file_key

check_and_reset_if_img_changed("old", file_key_old, w0, h0)
x0_left, x0_right, y0_top, y0_bottom = [roi_dict[file_key_old][k] for k in ["x_left","x_right","y_top","y_bottom"]]
st.sidebar.header("旧画像ROI（全体紫枠, 100刻み）")
x0_left   = st.sidebar.number_input("左端 x0",  0, w0-1, int(x0_left),   step=100, key="x0_left")
x0_right  = st.sidebar.number_input("右端 x0", 0, w0-1, int(x0_right),  step=100, key="x0_right")
y0_top    = st.sidebar.number_input("上端 y0", 0, h0-1, int(y0_top),    step=100, key="y0_top")
y0_bottom = st.sidebar.number_input("下端 y0", 0, h0-1, int(y0_bottom), step=100, key="y0_bottom")
xmin0, xmax0 = sorted([int(x0_left), int(x0_right)])
ymin0, ymax0 = sorted([int(y0_top), int(y0_bottom)])
roi_dict[file_key_old] = dict(x_left=x0_left, x_right=x0_right, y_top=y0_top, y_bottom=y0_bottom)

# --- 新画像についてはlast_used_new_roiを引き継ぐ！ ---
check_and_reset_if_img_changed("new", file_key_new, w1, h1)
if file_key_new in roi_dict:
    last_new_roi = roi_dict[file_key_new]
else:
    last_new_roi = st.session_state["last_used_new_roi"]
x1_left, x1_right, y1_top, y1_bottom = [last_new_roi[k] for k in ["x_left","x_right","y_top","y_bottom"]]
st.sidebar.header("新画像ROI（全体紫枠, 100刻み）")
x1_left   = st.sidebar.number_input("左端 x1",  0, w1-1, int(x1_left),   step=100, key="x1_left")
x1_right  = st.sidebar.number_input("右端 x1", 0, w1-1, int(x1_right),  step=100, key="x1_right")
y1_top    = st.sidebar.number_input("上端 y1", 0, h1-1, int(y1_top),    step=100, key="y1_top")
y1_bottom = st.sidebar.number_input("下端 y1", 0, h1-1, int(y1_bottom), step=100, key="y1_bottom")
xmin1, xmax1 = sorted([int(x1_left), int(x1_right)])
ymin1, ymax1 = sorted([int(y1_top), int(y1_bottom)])
roi_dict[file_key_new] = dict(x_left=x1_left, x_right=x1_right, y_top=y1_top, y_bottom=y1_bottom)

# 新画像のROI値を必ず維持
st.session_state["last_used_new_roi"] = roi_dict[file_key_new].copy()

# 画面表示用描画
def show_roi_on_full(img, x0, x1, y0, y1, color=(255,0,255), thick=24):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_rgb, (x0, y0), (x1, y1), color, thick)
    return img_rgb

c0, c1 = st.columns(2)
with c0: st.image(show_roi_on_full(img0, xmin0, xmax0, ymin0, ymax0), channels='BGR', caption="旧ROI全体+紫枠")
with c1: st.image(show_roi_on_full(img1, xmin1, xmax1, ymin1, ymax1), channels='BGR', caption="新ROI全体+紫枠")

roi0, roi1 = img0[ymin0:ymax0, xmin0:xmax0], img1[ymin1:ymax1, xmin1:xmax1]
c2, c3 = st.columns(2)
with c2: st.image(roi0, caption="旧ROI 拡大")
with c3: st.image(roi1, caption="新ROI 拡大")

# -----------------------------
# マーカ自動候補パラメータ(維持)UI
st.sidebar.header("マーカ自動候補パラメータ（維持）")
auto_param = st.session_state["marker_auto_params"]
auto_param["thresh"] = st.sidebar.number_input("しきい値", 0, 255, int(auto_param["thresh"]), step=5)
auto_param["area_min"] = st.sidebar.number_input("面積min", 1, 120, int(auto_param["area_min"]), step=1)
auto_param["area_max"] = st.sidebar.number_input("面積max", 10, 500, int(auto_param["area_max"]), step=10)
st.session_state["marker_auto_params"] = auto_param

def detect_markers(img, thresh, area_min, area_max):
    _, bw = cv2.threshold(255-img, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area_min <= area <= area_max:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                centers.append([cx, cy])
    return np.array(centers)

# クリック選択用だけ！確定可視化は絶対混在させない
cands0 = detect_markers(roi0, auto_param["thresh"], auto_param["area_min"], auto_param["area_max"])
cands1 = detect_markers(roi1, auto_param["thresh"], auto_param["area_min"], auto_param["area_max"])
roi0_for_click = cv2.cvtColor(roi0, cv2.COLOR_GRAY2RGB)
for i, (cx, cy) in enumerate(cands0):
    cv2.circle(roi0_for_click, (cx, cy), 15, (255,0,0), 4)
    cv2.putText(roi0_for_click, str(i), (cx+15, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)
roi1_for_click = cv2.cvtColor(roi1, cv2.COLOR_GRAY2RGB)
for i, (cx, cy) in enumerate(cands1):
    cv2.circle(roi1_for_click, (cx, cy), 15, (255,0,0), 4)
    cv2.putText(roi1_for_click, str(i), (cx+15, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)

old_pts_key = f"old_pts_{file_key_old}"
new_pts_key = f"new_pts_{file_key_new}"
if old_pts_key not in st.session_state["pts_dict"]: st.session_state["pts_dict"][old_pts_key] = []
if new_pts_key not in st.session_state["pts_dict"]: st.session_state["pts_dict"][new_pts_key] = []
if f"skip_flag_old_{file_key_old}" not in st.session_state: st.session_state[f"skip_flag_old_{file_key_old}"] = False
if f"skip_flag_new_{file_key_new}" not in st.session_state: st.session_state[f"skip_flag_new_{file_key_new}"] = False

col_clear_old, col_clear_new = st.columns(2)
with col_clear_old:
    if st.button("旧マーカー全消去"):
        st.session_state["pts_dict"][old_pts_key] = []
        st.session_state[f"skip_flag_old_{file_key_old}"] = True
with col_clear_new:
    if st.button("新マーカー全消去"):
        st.session_state["pts_dict"][new_pts_key] = []
        st.session_state[f"skip_flag_new_{file_key_new}"] = True

st.subheader("旧画像ROI：クリック＋手入力でマーカー指定")
oldclick = streamlit_image_coordinates(roi0_for_click, key="oldimg")
if oldclick:
    pt = [int(oldclick["x"]), int(oldclick["y"])]
    if not st.session_state[f"skip_flag_old_{file_key_old}"]:
        if (not st.session_state["pts_dict"][old_pts_key]) or (pt != st.session_state["pts_dict"][old_pts_key][-1]):
            st.session_state["pts_dict"][old_pts_key].append(pt)
if st.session_state[f"skip_flag_old_{file_key_old}"]:
    st.session_state[f"skip_flag_old_{file_key_old}"] = False
old_man = st.text_input("旧:手入力追加(X,Y)", key="old_man")
if st.button("旧:手入力追加") and old_man:
    try: x,y = [int(s.strip()) for s in old_man.split(",")]
    except: x,y = None,None
    if x is not None:
        pt = [x, y]
        if (not st.session_state["pts_dict"][old_pts_key]) or (pt != st.session_state["pts_dict"][old_pts_key][-1]):
            st.session_state["pts_dict"][old_pts_key].append(pt)

st.subheader("新画像ROI：クリック＋手入力でマーカー指定")
newclick = streamlit_image_coordinates(roi1_for_click, key="newimg")
if newclick:
    pt = [int(newclick["x"]), int(newclick["y"])]
    if not st.session_state[f"skip_flag_new_{file_key_new}"]:
        if (not st.session_state["pts_dict"][new_pts_key]) or (pt != st.session_state["pts_dict"][new_pts_key][-1]):
            st.session_state["pts_dict"][new_pts_key].append(pt)
if st.session_state[f"skip_flag_new_{file_key_new}"]:
    st.session_state[f"skip_flag_new_{file_key_new}"] = False
new_man = st.text_input("新:手入力追加(X,Y)", key="new_man")
if st.button("新:手入力追加") and new_man:
    try: x,y = [int(s.strip()) for s in new_man.split(",")]
    except: x,y = None,None
    if x is not None:
        pt = [x, y]
        if (not st.session_state["pts_dict"][new_pts_key]) or (pt != st.session_state["pts_dict"][new_pts_key][-1]):
            st.session_state["pts_dict"][new_pts_key].append(pt)

# 【確定表示：採用点のみ描画】
roi0_disp2 = cv2.cvtColor(roi0, cv2.COLOR_GRAY2RGB)
for i, (cx, cy) in enumerate(st.session_state["pts_dict"][old_pts_key]):
    cv2.circle(roi0_disp2, (cx, cy), 22, (255,0,0), 7)
    cv2.putText(roi0_disp2, f"{i}", (cx+18, cy-12), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,0), 5)
roi1_disp2 = cv2.cvtColor(roi1, cv2.COLOR_GRAY2RGB)
for i, (cx, cy) in enumerate(st.session_state["pts_dict"][new_pts_key]):
    cv2.circle(roi1_disp2, (cx, cy), 22, (255,0,0), 7)
    cv2.putText(roi1_disp2, f"N{i}", (cx+18, cy-12), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,0), 5)
cc2, cc3 = st.columns(2)
with cc2: st.image(roi0_disp2, channels='RGB', caption="旧ROIマーカー")
with cc3: st.image(roi1_disp2, channels='RGB', caption="新ROIマーカー")

st.header("ペア指定＆ひずみ計算")
A = np.array(st.session_state["pts_dict"][old_pts_key])
B = np.array(st.session_state["pts_dict"][new_pts_key])
if len(A) >= 2 and len(B) >= 2:
    optionsA = list(range(len(A)))
    optionsB = [f"N{i}" for i in range(len(B))]
    idxA1 = st.selectbox("旧画像:点番号A", optionsA, 0, key="pairA1")
    idxA2 = st.selectbox("旧画像:点番号B", optionsA, 1, key="pairA2")
    idxB1 = st.selectbox("新画像:点番号A", optionsB, 0, key="pairB1")
    idxB2 = st.selectbox("新画像:点番号B", optionsB, 1, key="pairB2")
    idxB1r = int(idxB1[1:]) if isinstance(idxB1, str) else idxB1
    idxB2r = int(idxB2[1:]) if isinstance(idxB2, str) else idxB2
    d0 = np.linalg.norm(A[idxA1] - A[idxA2])
    d1 = np.linalg.norm(B[idxB1r] - B[idxB2r])
    strain = (d1-d0)/d0
    st.success(f"【ひずみ計算】 旧{idxA1}-{idxA2} / 新{idxB1}-{idxB2}: 旧長={d0:.2f} 新長={d1:.2f} ひずみ={strain*100:.3f}%")
else:
    st.info("各ROIで2点以上マーカーを登録してください。")