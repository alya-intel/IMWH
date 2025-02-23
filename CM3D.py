import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# Configuración del modelo
brain_mesh = o3d.io.read_triangle_mesh("full_brain_binary.stl")
brain_mesh.compute_vertex_normals()  
mesh_center = brain_mesh.get_center()
brain_mesh.translate(-mesh_center)
brain_mesh.paint_uniform_color([0.9, 0.9, 0.9])

# Configurar visualizador
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720)
vis.add_geometry(brain_mesh)

# Configuración mínima de iluminación para macOS Metal
render_opt = vis.get_render_option()
render_opt.light_on = True  
render_opt.background_color = np.array([0.1, 0.1, 0.1])
render_opt.mesh_show_back_face = True
render_opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal


# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

class ControllerState:
    def __init__(self):
        self.translation = np.array([0.0, 0.0, -5.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.pinch_base = None

state = ControllerState()
smoothing = 0.15
smooth_rotation = np.array([0.0, 0.0, 0.0])
smooth_translation = np.array([0.0, 0.0, -5.0])
original_vertices = np.asarray(brain_mesh.vertices).copy()
original_triangles = np.asarray(brain_mesh.triangles).copy()

def get_pinch_strength(hand_landmarks):
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return np.linalg.norm([thumb.x - index.x, thumb.y - index.y])

def is_fist(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    return all(np.linalg.norm([hand_landmarks.landmark[t].x - palm.x, 
                              hand_landmarks.landmark[t].y - palm.y]) < 0.15 for t in tips)

def reset_to_original():
    global smooth_rotation, smooth_translation
    smooth_rotation = np.array([0.0, 0.0, 0.0])
    smooth_translation = np.array([0.0, 0.0, -5.0])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    reset_requested = False
    current_hands = {"left": None, "right": None}
    
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label.lower()
            current_hands[handedness] = hand
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if is_fist(hand): reset_requested = True
    else:
        reset_requested = True
    
    if reset_requested:
        reset_to_original()
    else:
        if current_hands["left"]:
            index = current_hands["left"].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            target_x = (index.x - 0.5) * 5
            target_y = -(index.y - 0.5) * 5
            smooth_translation[:2] = smooth_translation[:2] * 0.8 + np.array([target_x, target_y]) * 0.2
        
        if current_hands["right"]:
            index = current_hands["right"].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            target_rot_x = (index.y - 0.5) * 360
            target_rot_y = (index.x - 0.5) * 360
            smooth_rotation[:2] = smooth_rotation[:2] * 0.8 + np.array([target_rot_x, target_rot_y]) * 0.2
    
    # Actualizar geometría
    brain_mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
    brain_mesh.triangles = o3d.utility.Vector3iVector(original_triangles)
    brain_mesh.rotate(R.from_euler('xy', smooth_rotation[:2], degrees=True).as_matrix(), center=mesh_center)
    brain_mesh.translate(smooth_translation, relative=False)
    
    # Actualizar visualización
    vis.update_geometry(brain_mesh)
    vis.poll_events()
    vis.update_renderer()
    
    cv2.imshow('Neuro Control', cv2.resize(frame, (640, 480)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()